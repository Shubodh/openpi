"""
Tokenize both LIBERO-Goal task prompts with the PaliGemma SentencePiece tokenizer.

Run locally (needs sentencepiece + gcsfs, installs to /tmp if missing):
    python tmp_kv_cache_sanity_check/tokenize_prompts.py

Run on RunPod (tokenizer already cached after first run):
    cd /workspace/openpi
    source runpod/libero_env.sh  # or whichever venv has openpi installed
    python tmp_kv_cache_sanity_check/tokenize_prompts.py

Output: token ids + piece strings for each prompt, printed to stdout.
"""

import sys, os, warnings

# Allow running locally with sentencepiece/gcsfs pre-installed to /tmp/claude-1000/sp_install
_sp_path = '/tmp/claude-1000/sp_install'
if os.path.isdir(_sp_path):
    sys.path.insert(0, _sp_path)
warnings.filterwarnings('ignore')

import sentencepiece

_TOKENIZER_CACHE = '/tmp/claude/paligemma_tokenizer.model'

def _get_tokenizer() -> sentencepiece.SentencePieceProcessor:
    if not os.path.exists(_TOKENIZER_CACHE):
        print(f'Tokenizer not found at {_TOKENIZER_CACHE}, downloading from GCS...')
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem(token='anon')
            os.makedirs(os.path.dirname(_TOKENIZER_CACHE), exist_ok=True)
            fs.get('big_vision/paligemma_tokenizer.model', _TOKENIZER_CACHE)
            print(f'Downloaded: {os.path.getsize(_TOKENIZER_CACHE)} bytes')
        except ImportError:
            # On RunPod / openpi venv, use openpi's own download machinery
            from openpi.shared.download import maybe_download
            path = maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})
            return sentencepiece.SentencePieceProcessor(model_file=str(path))
    else:
        print(f'Tokenizer cached: {os.path.getsize(_TOKENIZER_CACHE)} bytes')
    return sentencepiece.SentencePieceProcessor(model_file=_TOKENIZER_CACHE)


def tokenize_prompt(sp: sentencepiece.SentencePieceProcessor, prompt: str) -> None:
    ids = sp.encode(prompt, add_bos=True)
    pieces = [sp.id_to_piece(i) for i in ids]
    print(f'\nPrompt : {repr(prompt)}')
    print(f'Tokens : {len(ids)} (including BOS)')
    for i, (tid, piece) in enumerate(zip(ids, pieces)):
        marker = '  <-- OBJECT NAME' if piece.lstrip('▁') in ('bowl', 'wine', 'bottle') else ''
        print(f'  [{i:3d}] id={tid:7d}  piece={repr(piece)}{marker}')


if __name__ == '__main__':
    sp = _get_tokenizer()
    print(f'Vocab size: {sp.vocab_size()}')

    prompts = [
        'put the bowl on top of the cabinet',
        'put the wine bottle on top of the cabinet',
    ]
    for p in prompts:
        tokenize_prompt(sp, p)