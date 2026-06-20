"""builderr · transcript ENSEMBLING module (standalone — does not touch router/ASR).

Merges the FAST (English-specialist, romanized) and HINGLISH (code-switch) transcripts
into one output that is more faithful than either alone:

  * technical terms      → canonical spelling, capitalization from the better source
  * Hindi words          → kept ROMANIZED (never Devanagari) — abhi/nahi/karlo, not अभी/नहीं
  * numbers / acronyms   → joined  (p 95→p95, g p t 4→GPT4, 2 fa→2FA, a w s→AWS)
  * English words        → higher-confidence source wins

Public API:
    merge_transcripts(fast_text, hinglish_text, fast_confidence, hinglish_confidence)
      -> {"merged_text": str, "merge_score": float, "token_sources": [(token, source), ...]}

Approach: token-level Needleman–Wunsch (global DP) alignment with a transliteration-aware
canonical key, then per-token source selection by token class. Pure standard library;
runs offline.
"""
from __future__ import annotations

import re
from typing import Optional

# =============================================================================
# Devanagari detection + transliteration (so "अभी" aligns with "abhi")
# =============================================================================
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def contains_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text or ""))


def is_hindi_token(token: str) -> bool:
    """True if the token contains any Devanagari codepoint (U+0900–U+097F).
    Such tokens come from the Hinglish ASR and must be kept verbatim — never
    romanized, transliterated, or replaced (अभी must stay अभी, not become 'abhi')."""
    return bool(_DEVANAGARI.search(token or ""))


def _is_deva_char(c: str) -> bool:
    return "ऀ" <= c <= "ॿ"


_DEVA_INDEP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
_DEVA_MATRA = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n", "ँ": "n", "ः": "h", "ॉ": "o",
}
_DEVA_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    "ड़": "r", "ढ़": "rh", "क़": "q", "ख़": "kh", "ग़": "g",
    "ज़": "z", "फ़": "f", "य़": "y",
}
_DEVA_DIGIT = {"०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
               "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"}
_VIRAMA = "्"
# common whole words → exact romanization (overrides char transliteration for quality)
_DEVA_WORD = {
    "नहीं": "nahi", "नहि": "nahi", "अभी": "abhi", "मत": "mat", "करो": "karo",
    "करना": "karna", "पहले": "pehle", "मेरा": "mera", "तुम": "tum", "अच्छा": "acha",
    "भाई": "bhai", "हाँ": "haan", "है": "hai", "क्या": "kya", "कैसे": "kaise",
    "कर": "kar", "लो": "lo", "कुछ": "kuch", "सब": "sab", "यार": "yaar",
}


def romanize(word: str) -> str:
    """Transliterate a Devanagari word to a Latin approximation."""
    if word in _DEVA_WORD:
        return _DEVA_WORD[word]
    out: list[str] = []
    chars = list(word)
    i, n = 0, len(chars)
    while i < n:
        c = chars[i]
        if c in _DEVA_CONS:
            base = _DEVA_CONS[c]
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt in _DEVA_MATRA:
                out.append(base + _DEVA_MATRA[nxt]); i += 2; continue
            if nxt == _VIRAMA:
                out.append(base); i += 2; continue
            out.append(base + "a"); i += 1; continue
        if c in _DEVA_INDEP:
            out.append(_DEVA_INDEP[c]); i += 1; continue
        if c in _DEVA_MATRA:
            out.append(_DEVA_MATRA[c]); i += 1; continue
        if c in _DEVA_DIGIT:
            out.append(_DEVA_DIGIT[c]); i += 1; continue
        if not _is_deva_char(c):
            out.append(c)
        i += 1
    return "".join(out)


# =============================================================================
# TECH_TERMS — canonical spelling/capitalization (>= 300 entries)
# =============================================================================
def _mk(*words: str) -> dict[str, str]:
    """canonical words keyed by their lowercase form."""
    return {w.lower(): w for w in words}


TECH_TERMS: dict[str, str] = {}
TECH_TERMS.update(_mk(
    # ---- cloud / infra ----
    "AWS", "GCP", "Azure", "S3", "EC2", "EKS", "ECS", "Lambda", "RDS", "DynamoDB",
    "CloudFront", "Route53", "IAM", "VPC", "SQS", "SNS", "CloudWatch", "Fargate",
    "EBS", "ELB", "ALB", "NLB", "Athena", "Redshift", "Aurora", "Glue", "Kinesis",
    "Firebase", "Heroku", "Vercel", "Netlify", "Cloudflare", "DigitalOcean", "Linode",
    "Terraform", "Ansible", "Pulumi", "Packer", "Vault", "Consul", "Nomad",
    # ---- containers / orchestration ----
    "Docker", "Kubernetes", "k8s", "kubectl", "Helm", "Pod", "Istio", "Envoy",
    "containerd", "Podman", "Compose", "Minikube", "Rancher", "OpenShift",
    # ---- CI/CD / VCS ----
    "Jenkins", "CircleCI", "TravisCI", "GitHub", "GitLab", "Bitbucket", "ArgoCD",
    "Git", "SVN", "Mercurial", "Actions", "Drone", "TeamCity", "Bamboo",
    # ---- languages ----
    "Python", "JavaScript", "TypeScript", "Golang", "Go", "Rust", "Java", "Kotlin",
    "Swift", "Ruby", "PHP", "Scala", "Elixir", "Erlang", "Haskell", "Clojure",
    "Perl", "Lua", "Dart", "Julia", "MATLAB", "Bash", "PowerShell", "COBOL",
    # ---- frameworks / libs ----
    "React", "Angular", "Vue", "Svelte", "Next.js", "Nuxt", "Node", "Deno", "Bun",
    "Django", "Flask", "FastAPI", "Spring", "Rails", "Express", "Laravel", "Symfony",
    "jQuery", "Redux", "Tailwind", "Bootstrap", "Webpack", "Vite", "Babel", "ESLint",
    "Pytest", "JUnit", "Jest", "Mocha", "Cypress", "Playwright", "Selenium",
    # ---- data / streaming / db ----
    "Kafka", "Redis", "Postgres", "PostgreSQL", "MySQL", "MariaDB", "MongoDB",
    "Cassandra", "Elasticsearch", "OpenSearch", "Solr", "Neo4j", "CockroachDB",
    "Snowflake", "Databricks", "Spark", "Hadoop", "Hive", "Presto", "Flink",
    "Airflow", "Dagster", "dbt", "RabbitMQ", "Pulsar", "Memcached", "SQLite",
    "ClickHouse", "InfluxDB", "TimescaleDB", "Pinecone", "Weaviate", "Milvus", "Qdrant",
    # ---- AI / ML ----
    "GPT", "GPT4", "GPT4o", "Claude", "Gemini", "Llama", "Mistral", "Qwen", "Whisper",
    "PyTorch", "TensorFlow", "Keras", "JAX", "CUDA", "cuDNN", "ONNX", "HuggingFace",
    "Transformers", "LangChain", "LlamaIndex", "OpenAI", "Anthropic", "Cohere",
    "LLM", "RAG", "RLHF", "LoRA", "embeddings", "tokenizer", "Parakeet", "Nemo",
    # ---- web / protocols / security ----
    "API", "REST", "GraphQL", "gRPC", "HTTP", "HTTPS", "WebSocket", "Webhook",
    "JSON", "XML", "YAML", "TOML", "CSV", "Protobuf", "OAuth", "JWT", "SAML",
    "SSL", "TLS", "CORS", "CSRF", "XSS", "DDoS", "VPN", "DNS", "CDN", "CIDR",
    "SSH", "FTP", "SMTP", "TCP", "UDP", "IP", "MQTT", "2FA", "MFA", "SSO", "RBAC",
    # ---- concepts / metrics / roles ----
    "PRD", "MVP", "SLA", "SLO", "SLI", "KPI", "OKR", "QPS", "RPS", "TPS",
    "p50", "p90", "p95", "p99", "latency", "throughput", "bandwidth", "uptime",
    "cache", "rollback", "rollout", "canary", "deploy", "deployment", "staging",
    "prod", "production", "hotfix", "backend", "frontend", "fullstack", "DevOps",
    "MLOps", "SRE", "QA", "UAT", "CRUD", "ORM", "SDK", "CLI", "GUI", "TUI",
    "UI", "UX", "CI", "CD", "PR", "MR", "DB", "OS", "VM", "GPU", "CPU", "RAM",
    "ROM", "SSD", "HDD", "IO", "RPC", "API", "ABI", "AST", "DOM", "CSS", "HTML",
    "SQL", "NoSQL", "ETL", "ELT", "OLAP", "OLTP", "ACID", "BASE", "CAP",
    # ---- tools / products ----
    "Jira", "Confluence", "Slack", "Notion", "Figma", "Linear", "Asana", "Trello",
    "Cursor", "Codex", "Copilot", "VSCode", "IntelliJ", "PyCharm", "Vim", "Neovim",
    "Postman", "Insomnia", "Datadog", "Grafana", "Prometheus", "Sentry", "PagerDuty",
    "Splunk", "Kibana", "Tableau", "Looker", "Metabase", "Segment", "Amplitude",
    "Stripe", "Twilio", "SendGrid", "Auth0", "Okta", "Salesforce", "HubSpot",
    # ---- misc engineering ----
    "regex", "endpoint", "middleware", "namespace", "mutex", "goroutine", "async",
    "await", "callback", "webhook", "cron", "daemon", "kernel", "sysadmin", "firewall",
    "load-balancer", "microservice", "monolith", "serverless", "idempotent",
))
# keep simple lowercase keys for the p-metrics so the merger normalizes "p 95"
for _p in ("p50", "p90", "p95", "p99"):
    TECH_TERMS[_p] = _p
_TECH_LOOKUP = {k.lower(): v for k, v in TECH_TERMS.items()}


# =============================================================================
# ROMANIZED_HINDI — distinctive romanized Hindi tokens (>= 500 entries)
# English-colliding words (to/do/the/is/me/main/so/hi/are/on/in/it/...) are EXCLUDED.
# =============================================================================
ROMANIZED_HINDI: set[str] = set("""
hai hain ho hu hoon hun tha thi raha rahi rahe hota hoti hote hone hoga hogi honge hua hui hokar
kar karo karu karun karna karne karta karti karte kiya kiye kare karenge karke karwao karlo
kardo kardiya kardena karwana karaya karwaya kara
de dena diya diye denge dega degi dijiye dedo dedena dekar dedunga
le lo lena liya liye lega legi lenge lekar lelo leliya leke
ja jao jana jaana jata jati jate gaya gayi gaye jayega jayegi jayenge jaunga jaungi jakar jaa
aa aao aana aaya aaye aayi aata aati aate aayega aayenge aakar
dekh dekho dekhna dekha dekhi dekhe dekhta dekhti dekhte dekhega dekhkar
bol bolo bolna bola boli bole bolta bolti bolte bolega bolkar
suno sunna suna suni sune sunta sunti sunte sunega sunkar
likh likho likhna likha likhe likhta likhega likhkar
padh padho padhna padha padhe padhta padhega padhkar
samajh samjho samajhna samjha samjhi samjhe samajhta samjhao samajhkar samjhe
ban banao banana banaya banaye banta banti banega banakar bante
mil milo milna mila mili mile milta milti milte milega milenge milke
rakh rakho rakhna rakha rakhe rakhta rakhega rakhkar
chal chalo chalna chala chale chalta chalti chalega chalkar chalao
ruk ruko rukna ruka ruke rukega rukja rukkar
uth utho uthna utha uthao uthega uthkar
baith baitho baithna baitha baithe baithkar
laga lagao lagna lagta lagti lagte lagega lagi lagao lagaya
soch socho sochna socha sochta sochega sochkar
kha khao khana khaya khate khaega khakar
piyo peena piya pite peeke
bhej bhejo bhejna bheja bheje bhejega
bata batao batana bataya bataye batata
pucho puchna poochna pucha puche
hata hatao hatana hataya
nikal nikalo nikalna nikala nikle
daal daalo daalna daala
khol kholo kholna khola
pakad pakdo pakadna pakda
chod chodo chhodo chhodna choda
ghuma ghumao ghumna ghuma
milao laao layi layega banwa banwao
mai mein mujhe meri mere mera tu tera teri tere tujhe tum tumhe tumhare tumhari tumne
aap aapka aapko aapki hum humein hamein humara hamara hamare hamari woh wo voh yeh ye
inka unka inko unko iska uska iski uski inke unke kisi kisne kisko koi kuch kuchh
sabko sabhi apna apni apne khud usne maine hamne tumlog humlog unko humne
kal aaj aj parso pehle pehla pehli baad fir phir ab tab jab kab yahan yaha wahan waha
kahan kaha idhar udhar kidhar andar bahar upar uper niche neeche saath sath bina tak
sirf bilkul zyada jyada thoda thodi bahut bohot bhot kafi kaafi jaldi dhire dheere turant
hamesha kabhi roz rozana dobara wapas waapas filhaal filhal abhitak tabtak jabtak abhi
nahi nahin nhi na kyu kyun kyon kyunki kyonki kyunke kaise kaisa kaisi kya kyaa kitna kitni
kitne kaun kon kaunsa konsa matlab agar warna lekin magar aur toh phirbhi par pe ki ka ke ko
se bhi jo jise jiska wajah yaani yani jaise jaisa jaisi waisa waise itna itni utna aisa aise aisi vaise
accha acha achha achhi achhe theek thik sahi galat bura buri naya nayi purana purani bada badi
chota choti chhota chhoti kaam baat baatein cheez cheezein log logo logon paisa paise ghar
din raat samay waqt saal mahina mahine dost behen beta beti didi bhaiya bhaiyya bhai sahab saheb
ji han ha hmm arre arrey oye chalo chal bas shayad sayad zaroor zarur jarur shukriya dhanyavad
namaste namaskar swagat kripya maaf maafi mast ekdum vagairah sach sacchi jhooth gussa khush
pareshan dhyan haan kuchh sab yaar wala wali wale waala waale jeena khaana sona uthna milna
acchi bahot bahaut sahi galti dikkat samasya mushkil aasan mushqil zaroori jaruri faltu bekaar
badhiya shaandar zabardast kamaal gajab bakwas bekar sahihai theekhai chaltahai
mummy papa dada dadi nana nani chacha chachi mama mami bua phupha sasur saas devar
khana paani chai doodh roti sabzi daal chawal namak mirch cheeni
ghar bahar gaon shaher mohalla galli sadak bazaar dukaan mandir masjid school college
subah shaam dopahar raat sawera andhera ujala garmi sardi barish dhoop hawa baadal
""".split())
# remove any accidental English-collision tokens
ROMANIZED_HINDI -= {"to", "do", "the", "is", "me", "main", "so", "hi", "are", "on", "in",
                    "it", "by", "of", "as", "at", "an", "we", "he", "my", "no", "be", "or",
                    "if", "a", "i", "us", "am", "sun", "pee", "hue", "ban"}


# =============================================================================
# tokenization + canonicalization
# =============================================================================
_PUNCT = ".,!?;:\"'()[]{}…—–-"


def _split_trail(tok: str) -> tuple[str, str]:
    """Return (core, trailing_punctuation)."""
    core = tok.rstrip(_PUNCT)
    trail = tok[len(core):] if len(core) < len(tok) else ""
    core = core.lstrip(_PUNCT)
    return core, trail


def _strip(tok: str) -> str:
    return tok.strip(_PUNCT)


def _fold(s: str) -> str:
    """Vowel-fold a romanized token for fuzzy matching across spelling variants."""
    s = s.lower()
    for a, b in (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u")):
        s = s.replace(a, b)
    return s


def canon(tok: str) -> str:
    """Canonical alignment key: tech→canonical, Devanagari→romanized, else vowel-folded."""
    core = _strip(tok)
    if not core:
        return ""
    low = core.lower()
    if low in _TECH_LOOKUP or re.fullmatch(r"p\d{2,3}", low):
        return _TECH_LOOKUP.get(low, low).lower()
    if contains_devanagari(core):
        return _fold(romanize(core))
    return _fold(low)


def classify(tok: str) -> str:
    """tech | hindi | number | english | other."""
    core = _strip(tok)
    if not core:
        return "other"
    low = core.lower()
    if low in _TECH_LOOKUP or re.fullmatch(r"p\d{2,3}", low):
        return "tech"
    if contains_devanagari(core):
        return "hindi"
    if low in ROMANIZED_HINDI:
        return "hindi"
    if any(ch.isdigit() for ch in core):
        return "number"
    return "english"


def _merge_acronym_tokens(tokens: list[str]) -> list[str]:
    """Join spaced acronyms / number metrics into canonical single tokens:
    'p 95'→'p95', 'g p t 4'→'GPT4', 'a w s'→'AWS', '2 fa'→'2FA'."""
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        # a run of "short" alphanumeric tokens is a candidate acronym
        j = i
        while j < n and len(_strip(tokens[j])) <= 2 and _strip(tokens[j]).isalnum():
            j += 1
        merged = False
        if j - i >= 2:
            for k in range(j, i + 1, -1):  # try longest run first
                cand = "".join(_strip(t) for t in tokens[i:k]).lower()
                alpha = "".join(c for c in cand if c.isalpha())
                resolved: Optional[str] = None
                if cand in _TECH_LOOKUP:
                    resolved = _TECH_LOOKUP[cand]
                elif re.fullmatch(r"p\d{2,3}", cand):
                    resolved = cand
                if resolved and 1 <= len(alpha) <= 6:
                    # absorb a trailing number-word/digit if it forms a known term
                    # (g p t four → gpt+4 → GPT4), so it aligns cleanly with the other side
                    if k == j and j < n:
                        d = _as_digit(_strip(tokens[j]))
                        if d and (cand + d) in _TECH_LOOKUP:
                            out.append(_TECH_LOOKUP[cand + d]); i = j + 1; merged = True; break
                    out.append(resolved)
                    i = k
                    merged = True
                    break
        if not merged:
            out.append(tokens[i])
            i += 1
    return out


_NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                 "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def _as_digit(tok: str):
    low = tok.lower()
    if low in _NUMBER_WORDS:
        return _NUMBER_WORDS[low]
    if tok.isdigit():
        return tok
    return None


def _tokenize(text: str) -> list[str]:
    toks = (text or "").split()
    return _merge_acronym_tokens(toks)


# =============================================================================
# alignment (Needleman–Wunsch over canonical keys)
# =============================================================================
def _align(a: list[str], b: list[str]) -> list[tuple[str, int, int]]:
    """Global alignment. Returns ops: ('M', i, j) match/sub, ('D', i, -1) a-only,
    ('I', -1, j) b-only."""
    ca = [canon(t) for t in a]
    cb = [canon(t) for t in b]
    na, nb = len(a), len(b)
    INF = float("inf")
    dp = [[0.0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        dp[i][0] = i
    for j in range(1, nb + 1):
        dp[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            sub = 0.0 if ca[i - 1] == cb[j - 1] else 1.0
            dp[i][j] = min(dp[i - 1][j - 1] + sub, dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0)
    # backtrack
    ops: list[tuple[str, int, int]] = []
    i, j = na, nb
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = 0.0 if ca[i - 1] == cb[j - 1] else 1.0
            if dp[i][j] == dp[i - 1][j - 1] + sub:
                ops.append(("M", i - 1, j - 1)); i -= 1; j -= 1; continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1.0:
            ops.append(("D", i - 1, -1)); i -= 1; continue
        ops.append(("I", -1, j - 1)); j -= 1
    ops.reverse()
    return ops


# =============================================================================
# per-token source selection
# =============================================================================
def _choose_tech(tf_core: str, th_core: str) -> tuple[str, str]:
    """Return (token, source). Canonical spelling; capitalization from the better source."""
    key = (tf_core or th_core).lower()
    canonical = _TECH_LOOKUP.get(tf_core.lower()) or _TECH_LOOKUP.get(th_core.lower())
    if canonical is None:  # p-metric reconstructed
        canonical = key
    if tf_core == canonical:
        return canonical, "fast"
    if th_core == canonical:
        return canonical, "hinglish"
    # neither source is canonically cased → keep spoken casing, prefer one with caps
    if th_core and any(c.isupper() for c in th_core) and not any(c.isupper() for c in tf_core):
        return th_core, "hinglish"
    if tf_core:
        return tf_core, "fast"
    return th_core, "hinglish"


def _choose_number(tf_core: str, th_core: str, cf: float, ch: float) -> tuple[str, str]:
    """Canonicalize a numeric token: prefer the side that actually has digits;
    tie → higher confidence."""
    fd = any(c.isdigit() for c in tf_core)
    hd = any(c.isdigit() for c in th_core)
    if fd and not hd:
        return tf_core, "fast"
    if hd and not fd:
        return th_core, "hinglish"
    return (tf_core, "fast") if cf >= ch else (th_core, "hinglish")


def _choose_pair(tf: str, th: str, cf: float, ch: float) -> tuple[str, str]:
    """Pick the output token + source for an aligned (fast, hinglish) pair.

    Faithfulness rules (the ensemble may only fix tech/numbers/spacing/caps):
      1/3. Devanagari on either side → keep the Hinglish token VERBATIM (never romanize).
      4/6. Romanized-Hindi word     → keep the Hinglish token (don't let fast override).
           tech                     → canonicalize spelling/casing.
           number                   → canonicalize digits.
           else (English)           → higher confidence wins.
    """
    tf_core, tf_tr = _split_trail(tf)
    th_core, th_tr = _split_trail(th)
    trail = tf_tr or th_tr

    # RULE 1 & 3 — any Devanagari involved: keep Hinglish exactly, never transliterate.
    if is_hindi_token(tf) or is_hindi_token(th):
        if th_core:
            return th_core + th_tr, "hinglish"
        return tf_core + tf_tr, "fast"      # only fast had it — still no romanizing

    classes = {classify(tf), classify(th)}

    # RULE 4 & 6 — Hindi (even romanized) stays with the faithful Hinglish source.
    if "hindi" in classes:
        if th_core:
            return th_core + trail, "hinglish"
        return tf_core + trail, "fast"
    if "tech" in classes:
        tok, src = _choose_tech(tf_core, th_core)
        return tok + trail, src
    if "number" in classes:
        tok, src = _choose_number(tf_core, th_core, cf, ch)
        return tok + trail, src
    # English / other → exact agreement keeps fast; otherwise higher confidence
    if canon(tf) == canon(th):
        return (tf_core + trail) if tf_core else (th_core + trail), "both"
    if cf >= ch:
        return tf_core + trail, "fast"
    return th_core + trail, "hinglish"


def _emit_single(tok: str, source: str) -> tuple[str, str]:
    """Render a token that exists in only one transcript (indel)."""
    core, trail = _split_trail(tok)
    # RULE 1 — Devanagari is kept verbatim, never romanized/transliterated.
    if is_hindi_token(tok):
        return core + trail, source
    cls = classify(tok)
    if cls == "tech":
        canonical = _TECH_LOOKUP.get(core.lower())
        if canonical is None and re.fullmatch(r"p\d{2,3}", core.lower()):
            canonical = core.lower()
        if canonical and (core == canonical or not any(c.isupper() for c in core)):
            return canonical + trail, source
        return core + trail, source
    return core + trail, source


# =============================================================================
# public API
# =============================================================================
def merge_transcripts(
    fast_text: str,
    hinglish_text: str,
    fast_confidence: float = 0.5,
    hinglish_confidence: float = 0.5,
) -> dict:
    """Merge fast + hinglish transcripts. Returns {merged_text, merge_score, token_sources}."""
    cf = max(0.0, min(1.0, float(fast_confidence)))
    ch = max(0.0, min(1.0, float(hinglish_confidence)))
    ft = (fast_text or "").strip()
    ht = (hinglish_text or "").strip()

    # degenerate cases — one side missing
    if not ft and not ht:
        return {"merged_text": "", "merge_score": 0.0, "token_sources": []}
    if not ht:
        toks = [_emit_single(t, "fast") for t in _tokenize(ft)]
        merged = _cleanup(toks)
        return {"merged_text": " ".join(t for t, _ in merged),
                "merge_score": round(0.5 * cf, 3), "token_sources": merged}
    if not ft:
        toks = [_emit_single(t, "hinglish") for t in _tokenize(ht)]
        merged = _cleanup(toks)
        return {"merged_text": " ".join(t for t, _ in merged),
                "merge_score": round(0.5 * ch, 3), "token_sources": merged}

    a = _tokenize(ft)
    b = _tokenize(ht)
    ops = _align(a, b)

    built: list[tuple[str, str, bool]] = []  # (token, source, is_solo)
    matches = 0
    aligned = 0
    for op, i, j in ops:
        if op == "M":
            aligned += 1
            if canon(a[i]) == canon(b[j]):
                matches += 1
            tok, src = _choose_pair(a[i], b[j], cf, ch)
            if _strip(tok):
                built.append((tok, src, False))
        elif op == "D":
            tok, src = _emit_single(a[i], "fast")
            if _strip(tok):
                built.append((tok, src, True))
        else:  # "I"
            tok, src = _emit_single(b[j], "hinglish")
            if _strip(tok):
                built.append((tok, src, True))

    merged = _cleanup(built)

    # merge_score: agreement between models blended with their confidences
    agreement = (matches / aligned) if aligned else 0.0
    avg_conf = (cf + ch) / 2.0
    merge_score = round(max(0.0, min(1.0, 0.6 * agreement + 0.4 * avg_conf)), 3)
    return {"merged_text": " ".join(t for t, _ in merged),
            "merge_score": merge_score, "token_sources": merged}


def _cleanup(built) -> list[tuple[str, str]]:
    """Drop solo romanized-Hindi fragments absorbed by an adjacent token (kar+lo→karlo),
    and collapse immediate duplicates. Accepts list of (tok, src) or (tok, src, solo)."""
    norm: list[tuple[str, str, bool]] = []
    for item in built:
        if len(item) == 3:
            norm.append(item)
        else:
            norm.append((item[0], item[1], False))

    out: list[tuple[str, str, bool]] = []
    for k, (tok, src, solo) in enumerate(norm):
        core = _strip(tok).lower()
        if solo and len(core) <= 3 and (core in ROMANIZED_HINDI):
            prev = _strip(out[-1][0]).lower() if out else ""
            nxt = _strip(norm[k + 1][0]).lower() if k + 1 < len(norm) else ""
            if (prev and (prev.endswith(core) or prev.startswith(core))) or \
               (nxt and (nxt.endswith(core) or nxt.startswith(core))):
                continue  # fragment already contained in a neighbor
        # collapse immediate duplicate (case-insensitive)
        if out and _strip(out[-1][0]).lower() == core and core:
            continue
        out.append((tok, src, solo))
    return [(t, s) for t, s, _ in out]


if __name__ == "__main__":
    # Faithfulness: Hinglish Devanagari is kept verbatim; the ensemble only fixes
    # tech/number/spacing/caps (p 95→p95, aws→AWS). It must NOT romanize अभी→abhi.
    cases = [
        ("rollback abhi mat karo pehle p 95 check kar lo",
         "rollback अभी मत करो पहले p95 check karlo",
         "rollback अभी मत करो पहले p95 check karlo"),
        ("deploy docker image to aws and rollback nahi karna",
         "deploy docker image to AWS and rollback नहीं करना",
         "deploy docker image to AWS and rollback नहीं करना"),
    ]
    ok = True
    for fast, hing, expected in cases:
        r = merge_transcripts(fast, hing, 0.7, 0.8)
        status = "PASS" if r["merged_text"] == expected else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{status}  got: {r['merged_text']!r}")
        if status == "FAIL":
            print(f"      exp: {expected!r}")
        print(f"      score={r['merge_score']}  sources={r['token_sources']}")
    print("\nTECH_TERMS:", len(TECH_TERMS), " ROMANIZED_HINDI:", len(ROMANIZED_HINDI))
    print("ALL EXAMPLES PASSED" if ok else "SOME EXAMPLES FAILED")
