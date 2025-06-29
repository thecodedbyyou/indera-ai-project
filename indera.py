# =============================================================================
# PROYEK INDERA++ (Intelligent Network-based Data Explorer and Repository Accumulator)
# Versi 10.0 - Edisi Heartbeat
# =============================================================================
# Deskripsi:
# Versi ini merombak total loop utama dan logging untuk transparansi dan ketahanan.
# 1. [HEARTBEAT LOG] Mengganti log yang berisik dengan satu baris ringkasan
#    status periodik yang informatif dan tidak menyebabkan lag.
# 2. [ANTI-STALL REFLECTION] Logika "Mode Refleksi" (belajar & berkreasi)
#    sekarang dijamin berjalan saat antrian benar-benar kosong.
# 3. [WIKIPEDIA IMMUNITY] INDERA dicegah mem-blacklist domain Wikipedia.
# 4. [MASSIVE CURRICULUM] Kurikulum awal diperluas secara drastis (~40 topik).
# =============================================================================

# @title Langkah 1: Instalasi, Impor, dan Mount Google Drive
def install_dependencies_and_mount_drive():
    """Menginstal semua library dan me-mount Google Drive."""
    print("🚀 Memulai instalasi library...")
    # !pip install -q requests beautifulsoup4 pandas numpy scikit-learn sentence-transformers spacy torch matplotlib joblib
    # !python -m spacy download en_core_web_sm
    print("✅ Instalasi selesai.")
    from google.colab import drive
    import os
    print("\n🔗 Meminta izin untuk mengakses Google Drive...")
    drive.mount('/content/drive', force_remount=True)
    drive_path = "/content/drive/My Drive/INDERA_Project_V10" # Folder baru untuk INDERA v10
    os.makedirs(drive_path, exist_ok=True)
    print(f"✅ Google Drive siap. File akan disimpan di: {drive_path}")
    return drive_path

DRIVE_PATH = install_dependencies_and_mount_drive()

# Impor semua library
import requests, spacy, time, sqlite3, hashlib, re, datetime, torch, os, shutil, joblib, random
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urljoin, urlparse, quote_plus
from urllib import robotparser
from collections import defaultdict
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError

# @title Konfigurasi dan Pemuatan Model Global
def load_models_and_config(drive_path):
    print("🧠 Memuat model NLP... Mengarahkan ke perangkat yang tersedia.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔌 Perangkat yang digunakan: {device.upper()}")
    if device == 'cpu': print("\n" + "!"*60 + "\n!!! PERINGATAN: GPU TIDAK AKTIF !!!\n" + "!"*60 + "\n")
    try: nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess; subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    config = {
        "DB_NAME": "indera_knowledge_base_v10.db", "DRIVE_PATH": drive_path,
        "LOCAL_DB_PATH": f"/content/indera_knowledge_base_v10.db",
        "DRIVE_DB_PATH": os.path.join(drive_path, 'indera_knowledge_base_v10.db'),
        "DRIVE_PLOT_PATH": os.path.join(drive_path, 'indera_progress.png'),
        "DRIVE_KEYWORD_MODEL_PATH": os.path.join(drive_path, 'keyword_quality_model.joblib'),
        "USER_AGENT": "INDERA-AI-Researcher/10.0 (Ethical Crawler; +http://your-project-url.com)",
        "TRAINING_INTERVAL_PAGES": 100, "TRAINING_EPOCHS": 5,
        "HEARTBEAT_INTERVAL": 10, # [HEARTBEAT LOG] Tampilkan ringkasan setiap 10 siklus
        "GRADUATION_REWARD_THRESHOLD": 20000, "GRADUATION_QUALITY_THRESHOLD": 0.68,
        "EPSILON_FOR_EXPLORATION": 0.15,
        "DOMAIN_ERROR_THRESHOLD": 5, "CRAWL_DELAY": 1,
    }
    print("✅ Model dan konfigurasi berhasil dimuat.")
    return nlp, similarity_model, config

nlp, similarity_model, CONFIG = load_models_and_config(DRIVE_PATH)
session_log_data = []

# @title [INDERA++] Otak Penilai Keyword
class KeywordQualityModel:
    def __init__(self):
        self.vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False)
        self.model = SGDRegressor(random_state=42, warm_start=True)
    def train(self, keywords, scores, epochs=1):
        X = self.vectorizer.transform(keywords); y = np.array(scores)
        for _ in range(epochs): self.model.partial_fit(X, y)
    def predict(self, keywords):
        try:
            X = self.vectorizer.transform(keywords); predictions = self.model.predict(X)
            return np.clip(predictions, 1, 10)
        except NotFittedError: return np.full(len(keywords), 5.0)
    def save(self, filepath): joblib.dump(self, filepath)
    @staticmethod
    def load(filepath):
        if os.path.exists(filepath):
            print(f"🧠 Menemukan dan memuat otak INDERA++ dari: {filepath}"); return joblib.load(filepath)
        print("🤔 Otak INDERA++ tidak ditemukan. Membuat yang baru."); return KeywordQualityModel()

# @title Langkah 2: Fungsi Utilitas
def setup_database_from_drive():
    if os.path.exists(CONFIG['LOCAL_DB_PATH']): os.remove(CONFIG['LOCAL_DB_PATH'])
    if os.path.exists(CONFIG['DRIVE_DB_PATH']):
        print(f"💾 Menemukan database di Drive. Menyalin..."); shutil.copy(CONFIG['DRIVE_DB_PATH'], CONFIG['LOCAL_DB_PATH'])
    else: print("⚠️ Database tidak ditemukan. Membuat yang baru.")
    conn = sqlite3.connect(CONFIG['LOCAL_DB_PATH']); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pages (id INTEGER PRIMARY KEY, url TEXT UNIQUE, content_hash TEXT, clean_text TEXT, quality_score REAL, reward REAL, source_keyword TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS url_queue (id INTEGER PRIMARY KEY, url TEXT UNIQUE, priority REAL, source_keyword TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS keyword_queue (id INTEGER PRIMARY KEY, keyword TEXT UNIQUE, priority REAL, depth INTEGER, parent_id INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS training_data (keyword TEXT PRIMARY KEY, total_score REAL DEFAULT 0, count INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS domain_blacklist (domain TEXT PRIMARY KEY, reason TEXT)''')
    conn.commit(); conn.close()
    print("✅ Database siap digunakan.")

def save_state_to_drive(keyword_model):
    print(f"\n💾 MENYIMPAN STATUS ({datetime.datetime.now().strftime('%H:%M:%S')}) 💾")
    keyword_model.save(CONFIG['DRIVE_KEYWORD_MODEL_PATH'])
    try:
        if os.path.exists(CONFIG['LOCAL_DB_PATH']):
            with sqlite3.connect(CONFIG['LOCAL_DB_PATH']) as s, sqlite3.connect(CONFIG['DRIVE_DB_PATH']) as d: s.backup(d)
            print(f"   -> ✅ Database disimpan.")
    except Exception as e: print(f"   -> ❌ Gagal menyimpan database: {e}")

def visualize_progress():
    print("\n📊 Membuat visualisasi progres sesi...")
    if not session_log_data: print("   -> Tidak ada data log."); return
    df = pd.DataFrame(session_log_data); fig, axes = plt.subplots(3, 1, figsize=(12, 18)); fig.suptitle('Progres Sesi INDERA++', fontsize=16)
    axes[0].plot(df['cycle'], df['total_reward'], label='Akumulasi Reward', c='b', marker='o', ms=2, alpha=0.7); axes[0].set_title('Akumulasi Reward'); axes[0].grid(True, alpha=0.6); axes[0].legend()
    axes[1].hist(df['quality_score'].dropna(), bins=20, color='g', alpha=0.7); axes[1].set_title('Distribusi Skor Kualitas'); axes[1].grid(True, alpha=0.6)
    axes[2].plot(df['cycle'], df['new_keywords_found'].cumsum(), label='Total Keyword', c='r', marker='x', ms=2, alpha=0.7); axes[2].set_title('Pertumbuhan Pengetahuan'); axes[2].grid(True, alpha=0.6); axes[2].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(CONFIG['DRIVE_PLOT_PATH']); print(f"   -> ✅ Grafik disimpan."); plt.show()

setup_database_from_drive()

# @title Langkah 3: Modul Inti (Class InderaCore)
class InderaCore:
    def __init__(self, db_path): self.db_path = db_path; self.robot_parsers = {}; self.domain_error_counts = defaultdict(int); self.domain_blacklist = self._load_blacklist()
    def _load_blacklist(self):
        with self.get_db_connection() as c: return {r[0] for r in c.execute("SELECT domain FROM domain_blacklist").fetchall()}
    def get_db_connection(self): return sqlite3.connect(self.db_path)
    def crawl_page(self, url):
        domain = urlparse(url).netloc
        if domain in self.domain_blacklist: return None, "domain_blacklisted", -1
        if not self._can_fetch(url): return None, "robots_denied", 0
        try:
            h = {'User-Agent': CONFIG['USER_AGENT'], 'Accept-Language': 'en-US,en;q=0.5'}
            with requests.Session() as s:
                r = s.get(url, headers=h, timeout=15); r.raise_for_status()
                if "text/html" in r.headers.get("Content-Type", ""): self.domain_error_counts[domain] = 0; return r.text, "success", r.status_code
                return None, "non_html", -1
        except requests.exceptions.RequestException as e:
            self.domain_error_counts[domain] += 1
            if self.domain_error_counts[domain] >= CONFIG['DOMAIN_ERROR_THRESHOLD']: self._blacklist_domain(domain, f"Error: {type(e).__name__}")
            reward = -2 if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404 else -3
            return None, "request_error", reward
    def _can_fetch(self, url):
        p_url = urlparse(url); d = f"{p_url.scheme}://{p_url.netloc}";
        if d not in self.robot_parsers: rp = robotparser.RobotFileParser(urljoin(d, 'robots.txt')); rp.read(); self.robot_parsers[d] = rp
        return self.robot_parsers[d].can_fetch(CONFIG['USER_AGENT'], url)
    def _blacklist_domain(self, domain, reason):
        # [WIKIPEDIA IMMUNITY] Jangan pernah blacklist Wikipedia
        if "wikipedia.org" in domain: return
        if domain not in self.domain_blacklist: self.domain_blacklist.add(domain)
        with self.get_db_connection() as c: c.execute("INSERT OR IGNORE INTO domain_blacklist VALUES (?,?)", (domain, reason))
    def process_html(self, html):
        s = BeautifulSoup(html, 'html.parser'); [el.decompose() for el in s(["script","style","nav","footer","header","aside"])]
        return '\n'.join(c for c in (p.strip() for l in s.get_text().splitlines() for p in l.split("  ")) if c)
    def evaluate_content_quality(self, text):
        if not text or len(text) < 500: return 0.1
        score = 0.5 + 0.1*min(len(text)/5000,1) + 0.1*("research" in text or "study" in text) + 0.15*bool(re.search(r'\[\d+\]|\(.*?,\s*\d{4}\)', text)) - 0.2*(text.count('!')>10) - 0.15*bool(len(re.findall(r'[A-Z]{5,}',text))>10)
        return max(0, min(1, score))
    def extract_keywords_from_text(self, text):
        doc = nlp(text); kws = set()
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART', 'EVENT']: kws.add(ent.text.strip().lower())
        for chunk in doc.noun_chunks:
            if 1 < len(chunk.text.split()) < 5: kws.add(chunk.text.strip().lower())
        return list(kws)
    def get_avg_quality_score(self, N=100):
        with self.get_db_connection() as c:
            result = c.execute("SELECT AVG(quality_score) FROM pages ORDER BY id DESC LIMIT ?", (N,)).fetchone()
            return result[0] if result and result[0] is not None else 0
    def generate_keywords_from_internal_knowledge(self, limit=50):
        all_new_keywords = set()
        with self.get_db_connection() as c:
            rows = c.execute("SELECT clean_text FROM pages WHERE quality_score > 0.8 ORDER BY RANDOM() LIMIT ?", (limit,)).fetchall()
            for row in rows:
                kws = self.extract_keywords_from_text(row[0]); all_new_keywords.update(kws)
        return list(all_new_keywords)

# @title Langkah 4 & 5: Otak INDERA++ dengan Heartbeat
def _perform_training_cycle(keyword_model, core_instance, epochs, reason="berkala"):
    """Fungsi terpusat untuk menjalankan siklus pelatihan otak."""
    print(f"\n🧠 WAKTUNYA BERPIKIR: Memulai siklus pelatihan {reason}...")
    with core_instance.get_db_connection() as conn:
        rows = conn.execute("SELECT keyword, total_score, count FROM training_data WHERE count > 0").fetchall()
        if rows:
            kw_train = [r[0] for r in rows]; scores_train = [r[1]/r[2] for r in rows]
            print(f"   -> Melatih otak dengan {len(rows)} pengalaman baru selama {epochs} epoch...")
            keyword_model.train(kw_train, scores_train, epochs=epochs)
            print(f"   -> ✅ Otak berhasil dilatih."); conn.execute("DELETE FROM training_data")
        else: print("   -> 🤔 Tidak ada pengalaman baru untuk dilatih saat ini.")

def run_indera():
    core = InderaCore(CONFIG['LOCAL_DB_PATH'])
    keyword_model = KeywordQualityModel.load(CONFIG['DRIVE_KEYWORD_MODEL_PATH'])
    total_reward = 0; last_save_time = time.time(); pages_crawled_this_session = 0; cycle = 0
    exploration_mode = 'WIKIPEDIA_ONLY'

    with core.get_db_connection() as conn:
        if conn.execute("SELECT COUNT(*) FROM keyword_queue").fetchone()[0]==0 and conn.execute("SELECT COUNT(*) FROM url_queue").fetchone()[0]==0:
            print("📚 Antrian kosong. Menginisialisasi dengan Kurikulum Awal yang Diperluas..."); 
            curriculum = [("history of artificial intelligence",10), ("deep learning",10), ("reinforcement learning",10), ("natural language processing",10), ("computer vision",9), ("transformer model",10), ("large language model",10), ("neural network",9), ("data structures and algorithms",8), ("cryptography",8), ("quantum computing",9), ("general relativity",9), ("string theory",8), ("crispr gene editing",10), ("dna sequencing",9), ("human genome project",9), ("theory of evolution",8), ("photosynthesis",8), ("mitochondria",8), ("industrial revolution",7), ("cognitive bias",8), ("stoicism",7), ("roman empire",7), ("behavioral economics",8), ("macroeconomics",7), ("geopolitics", 8), ("climate change", 9), ("renewable energy", 9), ("blockchain", 8), ("philosophy of mind", 7), ("ancient civilizations", 7), ("particle physics", 9), ("neuroscience", 10), ("psychology", 8), ("sociology", 7), ("art history", 6), ("music theory", 6)]
            kw_data = [(kw,prio,0,None) for kw,prio in curriculum]
            conn.executemany("INSERT OR IGNORE INTO keyword_queue(keyword,priority,depth,parent_id) VALUES(?,?,?,?)", kw_data); print(f"✅ {len(kw_data)} keyword awal ditambahkan.")
    
    print("\n" + "="*50 + "\n🤖 INDERA++ V10.0 Mulai Beroperasi 🤖\n" + "="*50 + "\n")
    try:
        while True:
            cycle += 1
            if time.time() - last_save_time > 60: save_state_to_drive(keyword_model); last_save_time = time.time()
            if pages_crawled_this_session >= CONFIG['TRAINING_INTERVAL_PAGES']:
                _perform_training_cycle(keyword_model, core, CONFIG['TRAINING_EPOCHS'])
                pages_crawled_this_session = 0
            
            with core.get_db_connection() as conn:
                if random.random() < CONFIG['EPSILON_FOR_EXPLORATION'] and exploration_mode == 'FULL_WEB':
                    task_source = "Eksplorasi Acak"
                    kw_task = conn.execute("SELECT id,keyword,priority,depth FROM keyword_queue ORDER BY priority ASC, id ASC LIMIT 1").fetchone()
                    url_task = None
                else:
                    task_source = "Prioritas Utama"
                    url_task = conn.execute("SELECT id,url,priority,source_keyword FROM url_queue ORDER BY priority DESC,id ASC LIMIT 1").fetchone()
                    kw_task = None if url_task else conn.execute("SELECT id,keyword,priority,depth FROM keyword_queue ORDER BY depth DESC,priority DESC,id ASC LIMIT 1").fetchone()

            if not url_task and not kw_task:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🏁 Antrian kosong. Memasuki Mode Refleksi...")
                _perform_training_cycle(keyword_model, core, CONFIG['TRAINING_EPOCHS'], reason="refleksi")
                newly_generated_kws = core.generate_keywords_from_internal_knowledge()
                if newly_generated_kws:
                    pred_prios = keyword_model.predict(newly_generated_kws)
                    kws_add = [(kw,p,0,None) for kw,p in zip(newly_generated_kws,pred_prios)]
                    with core.get_db_connection() as c: c.executemany("INSERT OR IGNORE INTO keyword_queue(keyword,priority,depth,parent_id) VALUES(?,?,?,?)", kws_add)
                    print(f"   -> ✅ Menemukan {len(kws_add)} ide riset baru. Melanjutkan eksplorasi.")
                time.sleep(10); continue

            cycle_reward, new_kws_cyc, avg_qual_cyc = 0, 0, 0
            
            if url_task:
                url_id, url, prio, source_keyword = url_task
                with core.get_db_connection() as c: c.execute("DELETE FROM url_queue WHERE id=?",(url_id,))
                html, stat, cr_rew = core.crawl_page(url); cycle_reward += cr_rew
                if stat == "success" and html:
                    txt = core.process_html(html); q_score = core.evaluate_content_quality(txt)
                    avg_qual_cyc = q_score; pages_crawled_this_session += 1
                    if source_keyword:
                        with core.get_db_connection() as c: c.execute("INSERT INTO training_data(keyword,total_score,count) VALUES(?,?,1) ON CONFLICT(keyword) DO UPDATE SET total_score=total_score+?,count=count+1",(source_keyword, q_score, q_score))
                    with core.get_db_connection() as conn:
                        cursor = conn.cursor()
                        c_hash = hashlib.sha256(txt.encode('utf-8')).hexdigest()
                        if cursor.execute("SELECT id FROM pages WHERE content_hash=?",(c_hash,)).fetchone(): cycle_reward -= 5
                        else:
                            i_rew = q_score * 10 + 2; cycle_reward += i_rew
                            kws = core.extract_keywords_from_text(txt); new_kws_cyc = len(kws)
                            if new_kws_cyc > 0: cycle_reward += new_kws_cyc * 0.2
                            cursor.execute("INSERT OR IGNORE INTO pages(url,content_hash,clean_text,quality_score,reward,source_keyword) VALUES (?,?,?,?,?,?)", (url,c_hash,txt,q_score,i_rew,source_keyword))
                            page_id = cursor.lastrowid
                            if page_id and kws:
                                pred_prios = keyword_model.predict(kws)
                                kws_add = [(kw,p,1,page_id) for kw,p in zip(kws,pred_prios)]
                                cursor.executemany("INSERT OR IGNORE INTO keyword_queue(keyword,priority,depth,parent_id) VALUES(?,?,?,?)", kws_add)
            elif kw_task:
                kw_id, kw, prio, depth = kw_task
                if exploration_mode == 'WIKIPEDIA_ONLY': s_url = f"https://en.wikipedia.org/wiki/{kw.replace(' ', '_')}"
                else: s_url = f"https://html.duckduckgo.com/html/?q={quote_plus(kw)}"
                with core.get_db_connection() as c: c.execute("INSERT OR IGNORE INTO url_queue(url,priority,source_keyword) VALUES(?,?,?)",(s_url,prio,kw)); c.execute("DELETE FROM keyword_queue WHERE id=?",(kw_id,)); cycle_reward+=0.1

            total_reward += cycle_reward
            session_log_data.append({'cycle':cycle, 'total_reward':total_reward, 'quality_score':avg_qual_cyc if avg_qual_cyc > 0 else None, 'new_keywords_found':new_kws_cyc})

            if cycle % CONFIG['HEARTBEAT_INTERVAL'] == 0:
                with core.get_db_connection() as c: uq, kq = c.execute("SELECT COUNT(*) FROM url_queue").fetchone()[0], c.execute("SELECT COUNT(*) FROM keyword_queue").fetchone()[0]
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Cycle: {cycle} | Reward: {total_reward:.1f} | URL Q: {uq} | Key Q: {kq} | Mode: {exploration_mode}")

            if exploration_mode == 'WIKIPEDIA_ONLY' and cycle % 100 == 0:
                avg_q = core.get_avg_quality_score()
                if total_reward > CONFIG['GRADUATION_REWARD_THRESHOLD'] and avg_q > CONFIG['GRADUATION_QUALITY_THRESHOLD']:
                    _perform_training_cycle(keyword_model, core, 10, reason="akhir pra-kelulusan")
                    exploration_mode = 'FULL_WEB'; print("\n" + "🎓"*20 + "\n🎓 KELULUSAN! INDERA Cukup Pintar Untuk Menjelajah Web Penuh! 🎓\n" + "🎓"*20 + "\n")
    
    except (KeyboardInterrupt, SystemExit) as e: print(f"\n\n🛑 Sesi dihentikan: {e} 🛑")
    finally:
        print("\n" + "="*50 + f"\n🏆 Sesi Operasi Selesai. Reward Akhir: {total_reward} 🏆")
        print("Finalizing..."); save_state_to_drive(keyword_model); visualize_progress()
        print("✅ Semua kemajuan disimpan.\n" + "="*50)

if __name__ == '__main__':
    run_indera()
