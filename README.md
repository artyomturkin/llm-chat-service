# Embeddings Service

Implements OpenAI chat completion API.

Supported models:
- mosaicml/mpt-7b-chat

Port can be configured with `CHAT_SERVER_PORT` environment variable, default `8080`.

Container blocks online download of models. Models should be mapped to `/var/models`:
```
/var/models
├── EleutherAI
│   └── gpt-neox-20b
│       ├── config.json
│       ├── merges.txt
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.json
└── mosaicml
    └── mpt-7b-chat
        ├── adapt_tokenizer.py
        ├── attention.py
        ├── blocks.py
        ├── config.json
        ├── configuration_mpt.py
        ├── custom_embedding.py
        ├── flash_attn_triton.py
        ├── generation_config.json
        ├── hf_prefixlm_converter.py
        ├── meta_init_context.py
        ├── modeling_mpt.py
        ├── norm.py
        ├── param_init_fns.py
        ├── pytorch_model-00001-of-00002.bin
        ├── pytorch_model-00002-of-00002.bin
        ├── pytorch_model.bin.index.json
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── version.txt
```

Example usage:
```sh
curl --location 'http://localhost:8080/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "mosaicml/mpt-7b-chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
}'
```
