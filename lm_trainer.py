import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, GPT2Config, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed, TrainerCallback
from tokenizers import SentencePieceBPETokenizer
import logging
logging.basicConfig(level=logging.ERROR)
import json

os.environ['CURL_CA_BUNDLE'] = ''  # if SSL Error

class LMTrainer:
    def __init__(self,
                 output_dir,
                 model_name,
                 input_fp,
                 config_dict,
                 model_train=None,
                 from_hub=True,
                 # args below will be passed as kwargs (config['lmtrainer'])
                 seed=42,
                 vocab_size=None,#=20_000,
                 data_size_for_lm=None,#=100_000_000,
                 data_size_for_tokenizer=None,#=5_000_000,
                 target_num_toks_for_lm=None,#=200_000_000,
                 context_length=None,#=128,
                 save_every_n_words=None,#=20_000_000,
                 save_at_n_words=None,
                 layers_to_unfreeze=None,#=["transformer.wte.weight"],
                 ):
        if not from_hub and not os.path.isfile(input_fp):
            raise IOError('The data have to be loaded from HF hub or a local directory.')
        self.seed = seed
        self.output_dir = output_dir
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.corpus = input_fp.split(os.path.sep)[-1].split('.')[0]
        self.lang, self.corpus_type = self.corpus.split('-')
        self.data_size_for_lm = data_size_for_lm
        self.data_size_for_tokenizer = data_size_for_tokenizer
        self.target_num_toks_for_lm = target_num_toks_for_lm
        self.save_every_n_words = save_every_n_words
        self.save_at_n_words = save_at_n_words
        self.context_length = context_length
        self.layers_to_unfreeze = layers_to_unfreeze
        self.lang2code = {'arabic':'ar', 'chinese':'zh-Hans', 'english':'en',
                          'japanese':'ja', 'portuguese':'pt', 'spanish':'es'}
        self.lm_config_dict = config_dict['lm']
        self.trainer_config_dict = config_dict['lm_training']
        self.simplewiki_title_fp = os.path.join('data', 'simplewiki_titles.json')
        self.model_train = model_train

    def load_data(self):
        try:
            if self.corpus_type == 'cc100':
                data = load_dataset('cc100', self.lang2code[self.lang], streaming=True)
                self.data = data['train']
            elif self.corpus_type == 'simplewiki':
                data = load_dataset('rahular/simple-wikipedia', streaming=True)
                self.data = data['train']
            else:
                data = load_dataset(self.corpus_type, streaming=True)
                self.data = data['train']
        except:
            raise IOError(f"{self.corpus_type} is not yet supported as of now.")

    def batchify_for_tokenizer(self, batch_size=1_000):
        total = 0
        batch = []
        for sample in self.data:
            if total >= self.data_size_for_tokenizer:
                return
            text = sample['text'].strip('\n')
            if not text:
                continue
            batch.append(text)
            if len(batch) == batch_size:
                total += batch_size
                yield batch
                batch = []
            print(f"\rBatchifying... {round(total / self.data_size_for_tokenizer, 3) * 100}%", end="")
        print("\n")

    def train_tokenizer(self):
        tokenizer = SentencePieceBPETokenizer()
        batches = self.batchify_for_tokenizer()
        tokenizer.train_from_iterator(batches,
                                      vocab_size=self.vocab_size,
                                      min_frequency=2,
                                      special_tokens=['<|endoftext|>'])
        gpt_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer,
                                          model_max_length = self.context_length,
                                          special_tokens = ['<|endoftext|>'])
        return gpt_tokenizer

    def chunk_for_lm(self, tokenizer):
        if self.corpus_type == 'simplewiki':
            with open(self.simplewiki_title_fp) as f:
                simplewiki_titles = json.load(f)
                simplewiki_titles = set(simplewiki_titles)
        data = self.data
        num_toks = 0
        toks = []
        for sample in data:
            text = sample['text'].strip('\n').strip()
            if self.corpus_type == 'cc100':  # new document signaled by '\n\n'
                if not text:
                    toks.append(tokenizer.eos_token_id)
                    continue
            elif self.corpus_type == 'simplewiki':  # new document signaled by the title of the page
                if text in simplewiki_titles:
                    toks.append(tokenizer.eos_token_id)
                    continue

            outputs = tokenizer(
                text,
                truncation=False
            )
            toks.extend(outputs['input_ids'])
            print(f"\rTokenizing... {round(len(toks) / self.data_size_for_lm, 5) * 100}%", end="")
            if len(toks) >= self.data_size_for_lm:
                break
        if self.corpus_type == 'simplewiki':
            simplewiki_titles.clear()  # clear
        chunks = []
        for start_idx in range(0, self.data_size_for_lm, self.context_length):
            chunks.append(toks[start_idx:start_idx + self.context_length])
            print(f"\rChunking into sequences of length {self.context_length}... {round(start_idx / self.data_size_for_lm, 5) * 100}%",
                  end="")
        toks = []
        chunks = {'input_ids': chunks[:-1]}  # dropping the last sequence

        return DatasetDict({'train': Dataset.from_dict(chunks)})

    def train_lm(self):
        # Initialize logger
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        logger.setLevel(logging.INFO)
        set_seed(self.seed)

        # Load data
        logger.info("Loading data")
        self.load_data()

        # Train tokenizer
        logger.info("Training tokenizer")
        tokenizer = self.train_tokenizer()
        tokenizer_model_path = os.path.join(self.output_dir, f"{self.model_name}-GPT2TokenizerFast")
        if not os.path.exists(tokenizer_model_path):
            os.mkdir(tokenizer_model_path)
        tokenizer.save_pretrained(tokenizer_model_path)

        logger.info("Tokenizing data")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        datasets = self.chunk_for_lm(tokenizer)
        print(f'Length of train data={len(datasets["train"])}')
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        # Initialize trainer
        if not self.model_train:  # need to initialize the model
            logger.info("Initialising GPT-2 from scratch")

            config = GPT2Config(vocab_size=self.vocab_size,
                                n_ctx=self.context_length,
                                n_positions=self.context_length,
                                eos_token_id=tokenizer.eos_token_id,
                                **self.lm_config_dict,
                                )
            model = AutoModelForCausalLM.from_config(config)
        else:
            logger.info("Loading the pretrained GPT-2")
            model = AutoModelForCausalLM.from_pretrained(self.model_train)
            model.resize_token_embeddings(new_num_tokens=0)  # reset the embeddings
            model.resize_token_embeddings(new_num_tokens=self.vocab_size)  # set the embeddings to the L2 vocab size
            model = self.freeze(model)
        model_size = sum(t.numel() for t in model.parameters())
        print(f"Model parameter size: {model_size / 1000 ** 2:.1f}M parameters")
        trainable_params = sum(t.numel() for t in model.parameters() if t.requires_grad==True)
        print(f"Trainable parameter size: {trainable_params / 1000 ** 2:.1f}M parameters")
        toks_per_step = (
                self.trainer_config_dict['per_device_train_batch_size']*\
                self.trainer_config_dict['gradient_accumulation_steps']*\
                self.context_length
        )
        max_steps = int(self.target_num_toks_for_lm/toks_per_step)
        
        if self.save_at_n_words:
            class CustomSaveCallback(TrainerCallback):
                def __init__(self, steps_to_save):
                    self.steps_to_save = set(steps_to_save)

                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step in self.steps_to_save:
                        # Save the model
                        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                        os.makedirs(output_dir, exist_ok=True)
                        kwargs['model'].save_pretrained(output_dir)
                        kwargs['tokenizer'].save_pretrained(output_dir)
                        print(f"Model saved at step {state.global_step}")                                    

            training_args = TrainingArguments(
                report_to=None,
                output_dir=str(self.output_dir),
                overwrite_output_dir=True,
                fp16=True,
                do_train=True,
                do_eval=False,
                do_predict=False,
                max_steps = max_steps,
                save_steps = 1_000_000_000,  # avoiding saving besides the custom saving
                logging_steps = 1_000_000_000,
                **self.trainer_config_dict
                )

            save_at_n_steps = [int(int(n_word)/toks_per_step) for n_word in self.save_at_n_words]

            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                data_collator=data_collator,
                train_dataset=datasets['train'],
                callbacks=[CustomSaveCallback(save_at_n_steps)]
                )

        elif self.save_every_n_words:
            training_args = TrainingArguments(
                report_to=None,
                output_dir=str(self.output_dir),
                overwrite_output_dir=True,
                #fp16=True,
                do_train=True,
                do_eval=False,
                do_predict=False,
                max_steps = max_steps,
                save_steps=int(self.save_every_n_words/toks_per_step),
                logging_steps=int(self.save_every_n_words/toks_per_step),
                **self.trainer_config_dict
            )

            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                data_collator=data_collator,
                train_dataset=datasets['train']
                )

        else:
            raise IOError("No saving method specified.")

        trainer.train()
        trainer.save_model()  # Saves the tokenizer too

    def freeze(self, model):
        if not self.layers_to_unfreeze:
            print("No layers are selected - all layers will remain trainable")
            return model
        for name, param in model.named_parameters():
            if name not in self.layers_to_unfreeze:
                param.requires_grad = False
            else:
                print(f"- {name}")
        print('above layers are set to be trainable (and all other layers are frozen!)')
        return model