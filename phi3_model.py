import onnxruntime_genai as og

class Phi3:
    def __init__(self, input_text, model='cuda/cuda-int4-rtn-block-32', min_length=None, max_length=4096, 
                 do_sample=False, top_p=None, top_k=None, temperature=None, repetition_penalty=None):
        self.model = og.Model(model)
        self.tokenizer = og.Tokenizer(self.model)

        self.search_options = {
            "min_length": min_length,
            "max_length": max_length,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty
        }
        self.search_options = {k: v for k, v in self.search_options.items() if v is not None}

        self.input_text = input_text
        self.generator = self.init_generator()

    def init_generator(self):
        chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
        prompt = f'{chat_template.format(input=self.input_text)}'
        input_tokens = self.tokenizer.encode(prompt)
        
        params = og.GeneratorParams(self.model)
        params.set_search_options(**self.search_options)
        params.input_ids = input_tokens

        generator = og.Generator(self.model, params)
        return generator

    def generate_tokens(self):
        while not self.generator.is_done():
            self.generator.compute_logits()
            self.generator.generate_next_token()
            next_token = self.generator.get_next_tokens()[0]
            yield self.tokenizer.decode([next_token])