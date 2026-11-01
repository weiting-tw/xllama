"""Compare HF tokenizer ids with llama.cpp llama-tokenize ids, add_special_tokens=False on both sides."""
import json, subprocess, sys
from transformers import AutoTokenizer
tok_dir, gguf, binary = sys.argv[1:4]  # usage: tok_parity.py <tokenizer_dir> <enc.gguf> <llama-tokenize>
tok = AutoTokenizer.from_pretrained(tok_dir)
# the exact pieces build_sequence tokenizes
pieces = ["choice question: Which department should handle this?", "score question: How urgent is this?",
          "noul question: Does the user threaten to cancel or leave?", " billing: invoices, payments, refunds",
          " technical: bugs, outages, system errors", " other: everything else", " level 0: not urgent",
          " false: no, the statement does not hold", " true: yes, the statement holds",
          "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
          "您好，三月份的帳單被重複扣款兩次，請今天退還重複的金額，否則我們會取消訂閱。",
          "每次打開設定頁面 App 就會閃退，從上次更新之後就這樣。",
          '{"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411", "amount": 129.99}',
          "訂單編號 A-20261101-0042，金額 NT$12,345 元；客戶：王小明（VIP）",
          "  leading spaces,\ttabs\nand newlines\n\n  end  ", "emoji 🙂👍🏽 and café naïve résumé",
          "日本語のテキストと한국어 텍스트, also Ελληνικά and русский", "x" * 3 + "1234567890" * 3,
          "請問 ¿qué tal? — “quotes” ‘single’ …", "SELECT * FROM t WHERE a<>b && c||d;"]
ok = 0
for p in pieces:
    hf = tok(p, add_special_tokens=False)["input_ids"]
    out = subprocess.run([binary, "-m", gguf, "-p", p, "--ids", "--no-bos", "--no-parse-special", "--log-disable"],
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    lc = json.loads(out)
    same = hf == lc
    ok += same
    if not same:
        print("MISMATCH %r\n  hf=%s\n  lc=%s" % (p[:60], hf, lc))
print("%d/%d pieces identical" % (ok, len(pieces)))
