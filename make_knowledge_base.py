import os
import requests
import json
import pandas as pd
import time
from bs4 import BeautifulSoup

# -------------------------------
# بخش 1: پیش‌پردازش دیتاست‌ها بر اساس ستون‌های واقعی و استفاده شده در main.py
# -------------------------------

def preprocess_nsl_kdd(df):
    df = df.copy()
    df['description'] = df.apply(lambda row:
                                 f"Protocol: {row['protocol_type']}, Flag: {row['flag']}, Duration: {row['duration']:.2f}, "
                                 f"SrcBytes: {row['src_bytes']}, DstBytes: {row['dst_bytes']}, Count: {row['count']}, SrvCount: {row['srv_count']}",
                                 axis=1)
    return df[['description', 'label']]

def preprocess_cicids2017(df):
    df = df.copy()
    df['description'] = df.apply(lambda row:
        f"Flow Duration: {row['Flow Duration']:.2f}, "
        f"Fwd Packets: {row['Total Fwd Packets']:.2f}, "
        f"Bwd Packets: {row['Total Backward Packets']:.2f}, "
        f"Fwd Packet Length Max: {row['Fwd Packet Length Max']:.2f}, "
        f"Fwd Packet Length Mean: {row['Fwd Packet Length Mean']:.2f}, "
        f"Bwd Packet Length Max: {row['Bwd Packet Length Max']:.2f}, "
        f"Bwd Packet Length Mean: {row['Bwd Packet Length Mean']:.2f}, "
        f"Flow Bytes/s: {row['Flow Bytes/s']:.2f}, "
        f"Flow Packets/s: {row['Flow Packets/s']:.2f}", axis=1)
    return df[['description', 'Label']].rename(columns={'Label': 'label'})

def preprocess_unsw_nb15(df):
    df = df.copy()
    df['description'] = df.apply(lambda row:
        f"Duration: {row['dur']:.2f}, SrcBytes: {row['sbytes']:.2f}, DstBytes: {row['dbytes']:.2f}, "
        f"SrcTTL: {row['sttl']:.2f}, DstTTL: {row['dttl']:.2f}, SLoad: {row['Sload']:.2f}, DLoad: {row['Dload']:.2f}, "
        f"SPkts: {row['Spkts']:.2f}, DPkts: {row['Dpkts']:.2f}, TCP RTT: {row['tcprtt']:.2f}, SYNACK: {row['synack']:.2f}, ACKDAT: {row['ackdat']:.2f}",
        axis=1)
    return df[['description', 'attack_cat']].rename(columns={'attack_cat': 'attack_cat'})

# -------------------------------
# بخش 2: تولید اسناد حمله (توصیف عمومی) برای هر دیتاست
# -------------------------------

def generate_attack_documents_nsl(df, attack_label, max_samples=5):
    general_desc = {
        'normal': "Normal traffic with expected protocol and packet patterns.",
        'neptune': "TCP SYN flood attack causing denial of service by exhausting server resources.",
        'warezclient': "Client-side activity involving unauthorized software downloads or cracking attempts.",
        'ipsweep': "Network scanning attack to identify live hosts by sending ICMP echo requests.",
        'portsweep': "Scanning multiple ports on target hosts to find open services.",
        'teardrop': "Denial of service attack exploiting IP fragmentation vulnerabilities.",
        'nmap': "Use of Nmap tool for network discovery and security auditing via scanning.",
        'satan': "Automated vulnerability scanner performing reconnaissance on network hosts.",
        'smurf': "Denial of service attack using ICMP echo requests with spoofed source addresses.",
        'pod': "Ping of Death attack sending oversized ICMP packets to crash target systems.",
        'back': "Backdoor activity allowing unauthorized remote access to compromised systems.",
        'guess_passwd': "Brute force attack attempting to guess user passwords by repeated login attempts.",
        'ftp_write': "Exploitation of FTP write permissions to upload or modify files maliciously.",
        'multihop': "Use of multiple intermediary hosts to obfuscate attack origin and move laterally.",
        'rootkit': "Malicious software designed to hide presence and maintain privileged access.",
        'buffer_overflow': "Attack exploiting buffer overflow vulnerabilities to execute arbitrary code.",
        'imap': "Malicious use of IMAP protocol to access or manipulate email data.",
        'warezmaster': "Server-side activity related to distribution of unauthorized software or cracking tools.",
        'phf': "Exploitation of PHF CGI script vulnerabilities to execute arbitrary commands.",
        'land': "Denial of service attack by sending spoofed packets with identical source and destination addresses.",
        'loadmodule': "Loading and executing malicious modules to compromise system integrity.",
        'spy': "Spyware activity aimed at secretly gathering sensitive user information.",
        'perl': "Use of Perl scripting for executing malicious commands or automation of attacks."
    }

    documents = []
    if attack_label in general_desc:
        documents.append(general_desc[attack_label])
    attack_samples = df[df['label'] == attack_label]
    selected_samples = attack_samples.head(max_samples)
    for _, row in selected_samples.iterrows():
        documents.append(f"{row['description']} indicates {attack_label} attack.")
    return documents

def generate_attack_documents_cicids(df, attack_label, max_samples=5):
    general_desc = {
        'BENIGN': "Normal benign network traffic with no malicious activity.",
        'DDoS': "Distributed Denial of Service attack flooding the target with traffic.",
        'PortScan': "Scanning multiple ports to find open services.",
        'Bot': "Compromised host participating in botnet activities.",
        'Infiltration': "Unauthorized access attempt to infiltrate the network.",
        'Web Attack': "Attack targeting web applications and servers.",
        'Brute Force': "Repeated login attempts to guess passwords."
    }
    documents = []
    if attack_label in general_desc:
        documents.append(general_desc[attack_label])
    attack_samples = df[df['label'] == attack_label]
    selected_samples = attack_samples.head(max_samples)
    for _, row in selected_samples.iterrows():
        documents.append(f"{row['description']} indicates {attack_label} attack.")
    return documents

def generate_attack_documents_unsw(df, attack_label, max_samples=5):
    general_desc = {
        'Normal': "Normal network traffic with expected behavior.",
        'Exploits': "Exploitation of vulnerabilities to gain unauthorized access.",
        'DoS': "Denial of Service attacks targeting availability.",
        'Reconnaissance': "Scanning and probing to gather information.",
        'Backdoor': "Unauthorized backdoor access to systems.",
        'Shellcode': "Injection of malicious shellcode."
    }
    documents = []
    if attack_label in general_desc:
        documents.append(general_desc[attack_label])
    attack_samples = df[df['attack_cat'] == attack_label]
    selected_samples = attack_samples.head(max_samples)
    for _, row in selected_samples.iterrows():
        documents.append(f"{row['description']} indicates {attack_label} attack.")
    return documents

# -------------------------------
# بخش 3: بارگذاری داده‌های MITRE ATT&CK از فایل محلی
# -------------------------------

def load_mitre_techniques(file_path='data/enterprise-attack.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    techniques = {}
    for obj in data.get('objects', []):
        if obj.get('type') == 'attack-pattern':
            ext_refs = obj.get('external_references', [])
            tech_id = ''
            for ref in ext_refs:
                if ref.get('source_name') == 'mitre-attack':
                    tech_id = ref.get('external_id', '')
                    break
            if tech_id:
                name = obj.get('name', '')
                desc = obj.get('description', '')
                short_desc = desc.split('.')[0] if desc else ''
                techniques[tech_id] = {'name': name, 'description': short_desc}
    return techniques

def get_mitre_docs_for_attacks(attack_names, mitre_file_path='data/enterprise-attack.json'):
    attack_to_technique = {
        'normal': [],
        'neptune': ['T1498'],
        'warezclient': ['T1071'],
        'ipsweep': ['T1595'],
        'portsweep': ['T1595'],
        'teardrop': ['T1498'],
        'nmap': ['T1595'],
        'satan': ['T1595'],
        'smurf': ['T1498'],
        'pod': ['T1498'],
        'back': ['T1498'],
        'guess_passwd': ['T1110'],
        'ftp_write': ['T1106'],
        'multihop': ['T1021'],
        'rootkit': ['T1014'],
        'buffer_overflow': ['T1203'],
        'imap': ['T1071'],
        'warezmaster': ['T1071'],
        'phf': ['T1190'],
        'land': ['T1498'],
        'loadmodule': ['T1203'],
        'spy': ['T1081'],
        'perl': ['T1059'],
        'DDoS': ['T1498'],  # اضافه شده برای حملات DDoS

        # تاکتیک‌های کلی MITRE ATT&CK (در صورت نیاز می‌توانید تکنیک‌های مرتبط را اضافه کنید)
        'None': [],
        'Exploits': [],
        'Fuzzers': [],
        'Backdoor': ['T1210'],  # نمونه تکنیک Backdoor (مثلاً Remote Access Tools)
        'DoS': ['T1498'],  # Denial of Service
        'Generic': [],
        'Reconnaissance': ['T1595'],  # نمونه تکنیک اسکن و شناسایی
        'Shellcode': ['T1059'],  # اجرای کد مخرب
        'Analysis': [],
        'Worms': [],
    }

    techniques = load_mitre_techniques(mitre_file_path)

    mitre_docs = []
    for attack_name in attack_names:
        if attack_name in attack_to_technique:
            for tech_id in attack_to_technique[attack_name]:
                if tech_id in techniques:
                    t = techniques[tech_id]
                    mitre_docs.append(f"{tech_id}: {t['name']}. {t['description']}.")
                else:
                    mitre_docs.append(f"{tech_id}: Description not found.")
    return mitre_docs

# -------------------------------
# بخش 4: وب‌اسکرپینگ توضیحات CVE از سایت NVD
# -------------------------------

def scrape_cve_description(cve_id):
    url = f"https://nvd.nist.gov/vuln/detail/{cve_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CVE-Scraper/1.0)"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {cve_id}: HTTP {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        desc_p = soup.find('p', attrs={'data-testid': 'vuln-description'})
        if desc_p:
            desc_text = desc_p.get_text(strip=True)
        else:
            print(f"Description not found for {cve_id}")
            return None

        short_desc = desc_text.split('.')[0] + '.'
        return f"{cve_id}: {short_desc}"

    except Exception as e:
        print(f"Error scraping {cve_id}: {e}")
        return None

def scrape_multiple_cves(cve_list, delay=1.0):
    descriptions = []
    for cve_id in cve_list:
        desc = scrape_cve_description(cve_id)
        if desc:
            descriptions.append(desc)
        time.sleep(delay)
    return descriptions

# -------------------------------
# بخش 5: ساخت پایگاه دانش برای هر دیتاست
# -------------------------------

def build_knowledge_base_for_dataset(dataset_name, csv_path):
    print(f"Building knowledge base for {dataset_name}...")

    df = pd.read_csv(csv_path)

    if dataset_name.lower() == 'nsl-kdd':
        df_processed = preprocess_nsl_kdd(df)
        attack_types = df_processed['label'].unique()
        generate_docs_func = generate_attack_documents_nsl

    elif dataset_name.lower() == 'cicids2017':
        df_processed = preprocess_cicids2017(df)
        attack_types = df_processed['label'].unique()
        generate_docs_func = generate_attack_documents_cicids

    elif dataset_name.lower() == 'unsw':
        df_processed = preprocess_unsw_nb15(df)
        attack_types = df_processed['attack_cat'].unique()
        generate_docs_func = generate_attack_documents_unsw

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    kb_dir = f'data/knowledge_base/{dataset_name}'
    os.makedirs(kb_dir, exist_ok=True)

    with open(os.path.join(kb_dir, f"{dataset_name}.txt"), "w", encoding="utf-8") as f_kb:
        for attack in attack_types:
            docs = generate_docs_func(df_processed, attack_label=attack, max_samples=5)
            for doc in docs:
                f_kb.write(doc + "\n")
    print(f"Knowledge base for {dataset_name} created.")

    mitre_docs = get_mitre_docs_for_attacks(attack_types, mitre_file_path='data/enterprise-attack.json')
    with open(os.path.join(kb_dir, "mitre.txt"), "w", encoding="utf-8") as f_mitre:
        for doc in mitre_docs:
            f_mitre.write(doc + "\n")
    print(f"MITRE knowledge base for {dataset_name} created.")

    relevant_cves = {
        'normal': [],
        'neptune': ['CVE-1999-0532'],
        'warezclient': [],
        'ipsweep': [],
        'portsweep': [],
        'teardrop': ['CVE-1999-0007'],
        'nmap': [],
        'satan': [],
        'smurf': ['CVE-1999-0124'],
        'pod': ['CVE-1999-0010'],
        'back': ['CVE-2002-0649'],
        'guess_passwd': ['CVE-2001-0540'],
        'ftp_write': ['CVE-1999-0459'],
        'multihop': [],
        'rootkit': ['CVE-2003-0147'],
        'buffer_overflow': ['CVE-1999-0517'],
        'imap': ['CVE-2002-0649'],
        'warezmaster': [],
        'phf': ['CVE-1999-0002'],
        'land': ['CVE-1999-0144'],
        'loadmodule': ['CVE-1999-0003'],
        'spy': [],
        'perl': ['CVE-2000-0884'],
        'DDoS': [
            'CVE-2016-5696',
            'CVE-2018-0171',
            'CVE-2019-0708'
        ],

        # تاکتیک‌های کلی MITRE ATT&CK (تاکتیک‌ها معمولاً کلی هستند و CVE خاص ندارند؛ اینجا می‌توانید خالی بگذارید یا CVEهای مرتبط اضافه کنید)
        'None': [],
        'Exploits': [],  # می‌توانید CVEهای عمومی اکسپلویت‌ها اضافه کنید
        'Fuzzers': [],
        'Backdoor': ['CVE-2002-0649'],  # نمونه Backdoor
        'DoS': ['CVE-1999-0532', 'CVE-1999-0007'],  # نمونه‌های DoS
        'Generic': [],
        'Reconnaissance': [],  # معمولاً CVE ندارد چون بیشتر رفتار است
        'Shellcode': [],
        'Analysis': [],
        'Worms': [],
    }

    all_cves = []
    for cve_list in relevant_cves.values():
        all_cves.extend(cve_list)
    all_cves = list(set(all_cves))

    cve_docs = scrape_multiple_cves(all_cves)
    with open(os.path.join(kb_dir, "cve.txt"), "w", encoding="utf-8") as f_cve:
        for doc in cve_docs:
            f_cve.write(doc + "\n")
    print(f"CVE knowledge base for {dataset_name} created.")


if __name__ == "__main__":
    build_knowledge_base_for_dataset('nsl-kdd', 'data/NSL_KDD.csv')
    build_knowledge_base_for_dataset('cicids2017', 'data/CICIDS2017.csv')
    build_knowledge_base_for_dataset('unsw', 'data/UNSW.csv')
