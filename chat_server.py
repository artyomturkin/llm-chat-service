import transformers
import torch
import uuid
import time

# Setup MPT-7B-Chat model
name = 'mosaicml/mpt-7b-chat'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'torch'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!
config.max_seq_len = 4096

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.float16, # Load model weights in bfloat16
  trust_remote_code=True
)

model.to(device='cuda:0')
model.eval() # Evaluation mode is default, but calling it anyway

# Setup tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_end|>"])

# define custom stopping criteria object
class StopOnTokens(transformers.StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])

# Setup web server
from aiohttp import web

routes = web.RouteTableDef()

@routes.post('/v1/chat/completions')
async def mpt(request):
  b = await request.json()

  msgs = b.get("messages")
  if msgs == None:
    raise aiohttp.web.HTTPBadRequest
  
  prompt = ""
  for msg in msgs:
    role = msg.get("role")
    if role == "system":
      prompt += f"<|im_start|>system\n{msg.get('content')}\n<|im_end|>\n"
    if role == "user":
      prompt += f"<|im_start|>user\n{msg.get('content')}\n<|im_end|>\n"
    if role == "assistant":
      prompt += f"<|im_start|>assistant\n{msg.get('content')}\n<|im_end|>\n"

  prompt += "<|im_start|>assistant\n"
  max_tokens = b.get("max_tokens", 256)

  generate_params = {
    "max_new_tokens": max_tokens,
    "temperature": 0.5, 
    "top_p": 0.15, 
    "top_k": 0, 
    "use_cache": True, 
    "do_sample": True, 
    "stopping_criteria": stopping_criteria,
    "repetition_penalty": 1.1
  }

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  input_ids = input_ids.to(model.device)
  input_length = len(input_ids[0])

  generated_ids = model.generate(input_ids, **generate_params)
  gen_ids = generated_ids.cpu().tolist()[0][len(input_ids[0]):]
  generated_length = len(gen_ids)

  output = tokenizer.decode(gen_ids, skip_special_tokens=True)

  return web.json_response({
    'id': str(uuid.uuid4()),
    'created': time.time(),
    'object': 'chat.completion',
    'model': 'EleutherAI/gpt-neox-20b',
    'usage': {
        'prompt_tokens': input_length,
        'completion_tokens': generated_length,
        'total_tokens': generated_length + input_length
    },
    'choices': [{
        'index': 0,
        'finish_reason': 'stop',
        'message': {
          'role': 'assistant',
          'content': output
        }
    }]
  })

import os
port = os.getenv('CHAT_SERVER_PORT', 8080)

app = web.Application()
app.add_routes(routes)
web.run_app(app, port=port)
