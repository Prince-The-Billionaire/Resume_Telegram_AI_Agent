import os
import json
import io
import logging
import asyncio
from typing import Dict, List, Optional
import modal
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "google-genai", 
        "pypdf", 
        "weasyprint", 
        "pydantic", 
        "requests", 
        "beautifulsoup4", 
        "supabase", 
        "fastapi[standard]",
        "playwright",
        "httpx"
    )
    .run_commands("playwright install chromium", "playwright install-deps chromium")
    .apt_install("fonts-dejavu", "fonts-liberation", "fontconfig", "libglib2.0-0", "libcairo2", "libpango-1.0-0", "libpangocairo-1.0-0")
)

app = modal.App("z-node-career-agent", image=image)

# --- Pydantic Models ---
class PersonalInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    linkedin: str
    github: str

class SkillCategory(BaseModel):
    category_name: str
    subcategories: List[str]

class EducationEntry(BaseModel):
    degree: str
    institution: str
    grade: Optional[str] = None
    duration: str
    notable_project: Optional[str] = None

class JobEntry(BaseModel):
    company: str
    role: str
    duration: str
    achievements: List[str]  # Stored as narrative sentences

class ProjectEntry(BaseModel):
    title: str
    link: Optional[str] = None
    achievements: List[str]

class InterestCategory(BaseModel):
    label: str
    details: str

class HarvardResume(BaseModel):
    personal_info: PersonalInfo
    technical_skills: List[SkillCategory]
    education: List[EducationEntry]
    work_experience: List[JobEntry]
    key_projects: List[ProjectEntry]
    interests: List[InterestCategory]

class TechnicalGapInterrogator(BaseModel):
    needs_interview: bool
    questions: List[str]

class GitHubProjectInfo(BaseModel):
    title: str
    description: str
    live_link: Optional[str] = None
    achievements: List[str]

class GitHubAnalysisResult(BaseModel):
    top_projects: List[GitHubProjectInfo]

class QAPair(BaseModel):
    question: str
    recommended_answer: str

class ApplicationCheatSheet(BaseModel):
    match_score: int
    why_you_win: str
    likely_form_questions: List[QAPair]

class ScoredJob(BaseModel):
    title: str
    company: str
    link: str
    platform: str
    match_score: int
    why_fit: str

class JobRankerResult(BaseModel):
    top_matches: List[ScoredJob]

# --- CSS (No bullets for experience) ---
ENHANCED_RESUME_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  :root { --primary-color: #0f172a; --accent-color: #2563eb; --text-dark: #1e293b; --text-muted: #64748b; --border-color: #e2e8f0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', system-ui, sans-serif; color: var(--text-dark); background-color: #ffffff; line-height: 1.6; padding: 2rem; max-width: 850px; margin: 0 auto; }
  header { border-bottom: 2px solid var(--border-color); padding-bottom: 1.5rem; margin-bottom: 2rem; text-align: center; }
  header h1 { font-size: 2.25rem; font-weight: 700; color: var(--primary-color); letter-spacing: -0.025em; }
  .contact-info { display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin-top: 0.75rem; font-size: 0.875rem; color: var(--text-muted); }
  section { margin-bottom: 1.5rem; }
  section h2 { font-size: 1.15rem; font-weight: 600; color: var(--primary-color); text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.25rem; margin-bottom: 1rem; }
  .item { margin-bottom: 1.25rem; }
  .item-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.25rem; }
  .item-title { font-size: 1.05rem; font-weight: 600; color: var(--primary-color); }
  .item-subtitle { color: var(--accent-color); font-weight: 500; }
  .item-date { font-size: 0.85rem; color: var(--text-muted); font-weight: 400; }
  .starl-content { margin-top: 0.5rem; font-size: 0.95rem; color: #334155; line-height: 1.6; text-align: justify; }
  ul.bullet-points { list-style-type: disc; margin-left: 1.25rem; margin-top: 0.25rem; }
  ul.bullet-points li { margin-bottom: 0.25rem; font-size: 0.95rem; color: #334155; }
  .skills-container { margin-bottom: 0.5rem; font-size: 0.95rem; }
</style>
"""

# --- Services ---
class GeminiService:
    def __init__(self):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.types = types
        self.model = "gemini-2.5-flash"

    def _structured(self, prompt: str, schema, temp=0.2):
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=self.types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=temp)
        )
        return schema.model_validate_json(resp.text)

    def parse_master(self, raw_text: str) -> HarvardResume:
        prompt = f"Parse this raw text resume directly into the structured schema.\n\n{raw_text}"
        return self._structured(prompt, HarvardResume, 0.1)
        
    def generate_master_from_dictation(self, raw_dictation: str) -> HarvardResume:
        prompt = f"""You are an elite career agent converting unstructured dictation into a Master Resume. 
CRITICAL FORMATTING RULE: For work experience 'achievements', DO NOT use bullet points. Write 1 to 2 flowing narrative sentences using the STARL method (Situation, Task, Action, Result, Learning).
Dictation:\n{raw_dictation}"""
        return self._structured(prompt, HarvardResume, 0.3)

    def select_top_github_projects(self, repos_data: List[dict]) -> GitHubAnalysisResult:
        prompt = f"Analyze these GitHub repositories. Extract the top 3 strongest projects based on code complexity. Create strong narrative achievements for each.\nRepos:\n{json.dumps(repos_data, indent=2)}"
        return self._structured(prompt, GitHubAnalysisResult, 0.2)

    def gap_interview(self, master: HarvardResume, job_description: str) -> TechnicalGapInterrogator:
        prompt = f"""Compare the candidate's profile to the job description. Identify missing technical components.
If a gap exists, frame your question by referencing their PAST EXPERIENCE to prompt a STARL response. 
Example: 'I noticed you worked at Total Energies. Can you describe a time you used [Missing Skill] there to achieve a result, and what you learned?'
Resume:\n{master.model_dump_json()}\nJob Description:\n{job_description}"""
        return self._structured(prompt, TechnicalGapInterrogator, 0.3)

    def tailor_resume(self, master: HarvardResume, job_description: str, interview_qa: str, github_projects: List[ProjectEntry]) -> HarvardResume:
        prompt = f"""You are an expert career agent formatting a resume.
TAILORING RULES: 
1. Rewrite the work experience 'achievements'. Write 1 to 2 flowing narrative sentences per role strictly using the STARL method (Situation, Task, Action, Result, Learning). DO NOT use bullet points.
2. Intelligently weave critical missing keywords and technologies from the Job Description into the candidate's past roles based on their dictated answers. Ensure maximum ATS compatibility while making the integration look completely natural to a hiring manager.
3. Add the synthesized GitHub projects into the key_projects section.
Master Profile:\n{master.model_dump_json()}
GitHub Projects:\n{json.dumps([p.model_dump() for p in github_projects], indent=2)}
Job Description:\n{job_description}\nCandidate's Dictated Answers:\n{interview_qa}"""
        return self._structured(prompt, HarvardResume, 0.2)

    def rank_jobs(self, master: HarvardResume, scraped_jobs: List[Dict]) -> JobRankerResult:
        prompt = f"""Analyze these scraped jobs against the master resume.
Return ONLY the top 3-5 jobs where the candidate has the highest probability of passing ATS based on their specific stack.
Provide a 1-sentence 'why_fit' explanation.
Resume:\n{master.model_dump_json()}\nScraped Jobs:\n{json.dumps(scraped_jobs)}"""
        return self._structured(prompt, JobRankerResult, 0.3)

    def generate_cheat_sheet(self, master: HarvardResume, job_description: str) -> ApplicationCheatSheet:
        prompt = f"""Analyze this job description and resume. 
1. Score the match. 2. Explain why they win. 3. Predict the 3 most difficult custom application form questions and generate exact copy-paste answers.
Master Resume:\n{master.model_dump_json()}\nJob:\n{job_description}"""
        return self._structured(prompt, ApplicationCheatSheet, 0.3)

    def cover_letter(self, master: HarvardResume, job_description: str) -> str:
        prompt = f"Write a highly tailored, compelling cover letter based on this resume and job description. Ready to send without placeholder brackets.\n\nResume:\n{master.model_dump_json()}\n\nJob:\n{job_description}"
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return resp.text

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        prompt = "Transcribe this audio exactly as spoken."
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[self.types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"), prompt]
        )
        return resp.text

class StorageService:
    def __init__(self):
        from supabase import create_client
        self.client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    def get_or_create_profile(self, chat_id: int) -> dict:
        res = self.client.table("profiles").select("*").eq("chat_id", chat_id).execute()
        if res.data: return res.data[0]
        
        new_profile = {
            "chat_id": chat_id, "master_resume": {}, "current_state": "IDLE",
            "job_desc": "", "questions": [], "current_q_idx": 0, "qa_responses": "",
            "target_role": "", "target_location": "", "cron_enabled": True, "generate_buffer": "",
            "linkedin": "", "github": "", "github_projects": []
        }
        self.client.table("profiles").insert(new_profile).execute()
        return new_profile

    def update(self, chat_id: int, updates: dict):
        self.client.table("profiles").update(updates).eq("chat_id", chat_id).execute()

class ScraperService:
    def scrape_github_repos(self, github_url: str) -> List[dict]:
        import requests
        username = github_url.rstrip('/').split('/')[-1]
        api_url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=10"
        repos_data = []
        try:
            resp = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if resp.status_code == 200:
                for r in resp.json():
                    if not r.get("fork"):
                        repos_data.append({"name": r.get("name"), "description": r.get("description"), "language": r.get("language")})
        except Exception: pass
        return repos_data

    async def fetch_multi_platform_jobs(self, role: str, location: str) -> List[Dict]:
        from playwright.async_api import async_playwright
        results = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # 1. LinkedIn
                try:
                    await page.goto(f"https://www.linkedin.com/jobs/search/?keywords={role.replace(' ', '%20')}&location={location.replace(' ', '%20')}", timeout=15000)
                    cards = await page.query_selector_all("div.base-search-card")
                    for card in cards[:10]:
                        title = await (await card.query_selector(".base-search-card__title")).inner_text()
                        company = await (await card.query_selector(".base-search-card__subtitle")).inner_text()
                        link = await (await card.query_selector("a.base-card__full-link")).get_attribute("href")
                        results.append({"title": title.strip(), "company": company.strip(), "platform": "LinkedIn", "link": link.split('?')[0]})
                except Exception: pass
                
                # 2. RemoteOK 
                import requests
                try:
                    resp = requests.get(f"https://remoteok.com/api?tags={role.split()[0]}", headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code == 200:
                        for job in resp.json()[1:6]:
                            results.append({"title": job.get("position"), "company": job.get("company"), "platform": "RemoteOK", "link": job.get("url")})
                except Exception: pass
                
                await browser.close()
        except Exception as e:
            logger.error(f"Scraper error: {e}")
        return results

    async def extract_job_description_async(self, url: str) -> str:
        from playwright.async_api import async_playwright
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=20000)
                text = await page.inner_text("body")
                await browser.close()
                return ' '.join(text.split())[:4000]
        except Exception:
            return f"Context extraction failed for: {url}"

# --- Telegram Helpers ---
def tg_api(method: str, payload: dict = None, files: dict = None):
    import requests
    url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/{method}"
    resp = requests.post(url, json=payload if not files else None, data=payload if files else None, files=files)
    return resp.json()

def send_message(chat_id: int, text: str) -> int:
    res = tg_api("sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    return res.get("result", {}).get("message_id")

def edit_message(chat_id: int, msg_id: int, text: str):
    tg_api("editMessageText", {"chat_id": chat_id, "message_id": msg_id, "text": text, "parse_mode": "Markdown"})

def send_doc(chat_id: int, file_path: str, caption: str):
    with open(file_path, "rb") as f:
        tg_api("sendDocument", {"chat_id": chat_id, "caption": caption}, files={"document": f})

def download_tg_voice(file_id: str) -> bytes:
    import requests
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    file_info = tg_api("getFile", {"file_id": file_id})
    file_path = file_info["result"]["file_path"]
    resp = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}")
    return resp.content

def extract_pdf_text(file_id: str) -> str:
    import requests
    from pypdf import PdfReader
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    file_path = tg_api("getFile", {"file_id": file_id})["result"]["file_path"]
    resp = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}")
    reader = PdfReader(io.BytesIO(resp.content))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

def export_to_pdf(data: HarvardResume, output_filename="/tmp/resume.pdf"):
    from weasyprint import HTML
    skills = "".join([f"<div class='skills-container'><strong>{c.category_name}:</strong> {', '.join(c.subcategories)}</div>" for c in data.technical_skills])
    edu = "".join([f"<div class='item'><div class='item-header'><span class='item-title'>{e.institution}</span><span class='item-date'>{e.duration}</span></div><div class='item-subtitle'>{e.degree}</div></div>" for e in data.education])
    
    # Rendering Work Experience as a STARL narrative paragraph
    exp = "".join([f"<div class='item'><div class='item-header'><span class='item-title'>{j.company}</span><span class='item-date'>{j.duration}</span></div><div class='item-subtitle'>{j.role}</div><div class='starl-content'>{' '.join(j.achievements)}</div></div>" for j in data.work_experience])
    
    proj = "".join([f"<div class='item'><div class='item-header'><span class='item-title'>{p.title}</span></div><div class='item-subtitle'><a href='{p.link or '#'}'>{p.link or ''}</a></div><ul class='bullet-points'>{''.join([f'<li>{a}</li>' for a in p.achievements])}</ul></div>" for p in data.key_projects])
    
    html = f"""<html><head>{ENHANCED_RESUME_CSS}</head><body>
    <header><h1>{data.personal_info.name}</h1><div class='contact-info'><span>{data.personal_info.email}</span> | <span>{data.personal_info.linkedin}</span> | <span>{data.personal_info.github}</span></div></header>
    <section><h2>Technical Skills</h2>{skills}</section>
    <section><h2>Experience</h2>{exp}</section>
    <section><h2>Key Projects</h2>{proj}</section>
    <section><h2>Education</h2>{edu}</section>
    </body></html>"""
    HTML(string=html).write_pdf(output_filename)

# --- Bot Controller ---
class BotController:
    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.storage = StorageService()
        self.gemini = GeminiService()
        self.scraper = ScraperService()
        self.profile = {}

    async def handle_async(self, message: dict):
        text = message.get("text", "").strip() if "text" in message else ""
        
        self.profile = self.storage.get_or_create_profile(self.chat_id)
        state = self.profile.get("current_state", "IDLE")

        if text == "/start": return self._start()
        if text == "/stop": return self._stop_cron()
        if text == "/generate": return self._start_generate()
        if text in ["/scrape", "/newscrape"]: return self._ask_role()
        if text == "/tailor": return self._ask_tailor()
        if text == "/changeresume": return self._change_resume()
        if text == "/github": return self._trigger_github_sync()

        # Generation Flow
        if state.startswith("GENERATE_"):
            return self._process_generation(state, message)

        # File Upload Flow
        if state == "AWAITING_MASTER": return self._process_master(message)
        if state == "AWAITING_LINKEDIN": return self._process_linkedin(text)
        if state == "AWAITING_GITHUB": return self._process_github_initial(text)

        # Standard Machine
        if state == "AWAITING_GITHUB_SYNC": return self._process_github_sync(text)
        if state == "AWAITING_SCRAPE_ROLE": return self._process_role(text)
        if state == "AWAITING_SCRAPE_LOCATION": return await self._process_location(text)
        if state == "AWAITING_JOB_LINK": return await self._process_job_link(text)
        if state == "AWAITING_JOB_DESCRIPTION": return self._process_direct_job(text)
        if state == "INTERVIEW_MODE": return self._process_interview(message)
        if state == "AWAITING_COVER_LETTER_CONFIRM": return self._process_cover_letter(text)

        send_message(self.chat_id, "Command recognized. Type /start to see the full menu.")

    # --- Core Commands ---
    def _start(self):
        self.storage.update(self.chat_id, {"cron_enabled": True})
        
        welcome_msg = """🤖 **Welcome to your AI Career Agent!** 
        
Here is how I can help you land your next role:

🎙️ **/generate** - Build your Master Resume from scratch. Simply send **Voice Notes** or text to dictate your experience, and I will format it into a Harvard-style PDF.
🔍 **/scrape** (or **/newscrape**) - Tell me your target role and location. I will deploy scrapers (LinkedIn, RemoteOK, etc.) to return the highest-matching jobs.
✂️ **/tailor** - Paste a job description. I will interview you on your gaps and rewrite your resume using the **STARL** method to weave in keywords and beat the ATS.
🐙 **/github** - Connect your GitHub. I will scan your repos and automatically extract your top projects to inject into your resume.
📄 **/changeresume** - Upload a new Master Resume PDF.
🛑 **/stop** - Pause your daily automated job alerts.

Let's get to work! Upload a PDF as your Master Resume, or type /generate to start dictating."""

        send_message(self.chat_id, welcome_msg)
        
        if not self.profile.get("master_resume"):
            self.storage.update(self.chat_id, {"current_state": "AWAITING_MASTER"})

    def _stop_cron(self):
        self.storage.update(self.chat_id, {"cron_enabled": False})
        send_message(self.chat_id, "🛑 Daily alerts paused. Type /start to resume.")

    def _change_resume(self):
        self.storage.update(self.chat_id, {"current_state": "AWAITING_MASTER", "master_resume": {}})
        send_message(self.chat_id, "Upload your new Master Resume PDF.")

    # --- File Upload & Onboarding ---
    def _process_master(self, message):
        if "document" not in message or not message["document"].get("file_name", "").lower().endswith(".pdf"):
            send_message(self.chat_id, "Please upload a valid PDF document.")
            return
        send_message(self.chat_id, "Parsing architecture...")
        try:
            raw = extract_pdf_text(message["document"]["file_id"])
            parsed = self.gemini.parse_master(raw)
            self.storage.update(self.chat_id, {"master_resume": parsed.model_dump(), "current_state": "AWAITING_LINKEDIN"})
            send_message(self.chat_id, f"Parsed. Now, send your LinkedIn URL:")
        except Exception as e:
            send_message(self.chat_id, f"Parse error: {e}")

    def _process_linkedin(self, text):
        self.storage.update(self.chat_id, {"linkedin": text, "current_state": "AWAITING_GITHUB"})
        send_message(self.chat_id, "Received. Now send your GitHub URL:")

    def _process_github_initial(self, text):
        self.storage.update(self.chat_id, {"github": text, "current_state": "IDLE"})
        self._execute_github_scan(text)

    # --- GitHub Sync Flow ---
    def _trigger_github_sync(self):
        current_github = self.profile.get("github", "")
        if current_github:
            self._execute_github_scan(current_github)
        else:
            self.storage.update(self.chat_id, {"current_state": "AWAITING_GITHUB_SYNC"})
            send_message(self.chat_id, "No GitHub profile found. Please send your GitHub profile URL:")

    def _process_github_sync(self, text: str):
        self.storage.update(self.chat_id, {"github": text, "current_state": "IDLE"})
        self._execute_github_scan(text)

    def _execute_github_scan(self, github_url: str):
        msg_id = send_message(self.chat_id, "⏳ Deep-scanning GitHub for new projects...")
        raw_repos = self.scraper.scrape_github_repos(github_url)
        if raw_repos:
            analysis = self.gemini.select_top_github_projects(raw_repos)
            projects = [ProjectEntry(title=p.title, link=p.live_link, achievements=p.achievements).model_dump() for p in analysis.top_projects]
            self.storage.update(self.chat_id, {"github_projects": projects})
            edit_message(self.chat_id, msg_id, "✅ GitHub synced! Projects staged for your next application.")
        else:
            edit_message(self.chat_id, msg_id, "⚠️ No public repos found.")

    # --- Generation Flow ---
    def _start_generate(self):
        self.storage.update(self.chat_id, {"current_state": "GENERATE_PERSONAL", "generate_buffer": ""})
        send_message(self.chat_id, "Let's build your Master Resume. You can type or send **Voice Notes**.\n\nFirst, tell me your full name, email, phone, LinkedIn, and GitHub links.")

    def _process_generation(self, state: str, message: dict):
        text_input = self._extract_text_or_voice(message)
        if not text_input: return
        
        buffer = self.profile.get("generate_buffer", "") + f"\n[{state}]: {text_input}"
        
        if state == "GENERATE_PERSONAL":
            self.storage.update(self.chat_id, {"current_state": "GENERATE_EDU", "generate_buffer": buffer})
            send_message(self.chat_id, "Got it. Now dictate your Education (University, Degree, Graduation Date).")
        elif state == "GENERATE_EDU":
            self.storage.update(self.chat_id, {"current_state": "GENERATE_EXP", "generate_buffer": buffer})
            send_message(self.chat_id, "Great. Now list your Work Experience. Tell me the company, role, duration, and what you achieved (I will format it into a continuous STARL narrative).")
        elif state == "GENERATE_EXP":
            self.storage.update(self.chat_id, {"current_state": "GENERATE_SKILLS", "generate_buffer": buffer})
            send_message(self.chat_id, "Almost done. Dictate your Technical Skills (Languages, Frameworks, Tools).")
        elif state == "GENERATE_SKILLS":
            self.storage.update(self.chat_id, {"current_state": "IDLE", "generate_buffer": buffer})
            msg_id = send_message(self.chat_id, "⏳ Compiling and formatting your Master Resume...")
            master = self.gemini.generate_master_from_dictation(buffer)
            self.storage.update(self.chat_id, {"master_resume": master.model_dump()})
            pdf_path = "/tmp/Generated_Master.pdf"
            export_to_pdf(master, pdf_path)
            send_doc(self.chat_id, pdf_path, "✅ Master Resume Generated and saved to database!")
            edit_message(self.chat_id, msg_id, "Done.")

    def _extract_text_or_voice(self, message: dict) -> str:
        if "voice" in message:
            try:
                audio_bytes = download_tg_voice(message["voice"]["file_id"])
                text = self.gemini.transcribe_audio(audio_bytes)
                send_message(self.chat_id, f"📝 *Transcript:* {text}")
                return text
            except Exception:
                send_message(self.chat_id, "Voice transcription failed. Please type.")
                return None
        return message.get("text", "")

    # --- Scraping Flow ---
    def _ask_role(self):
        self.storage.update(self.chat_id, {"current_state": "AWAITING_SCRAPE_ROLE"})
        send_message(self.chat_id, "Target Role? (e.g., Python Developer)")

    def _process_role(self, text):
        self.storage.update(self.chat_id, {"target_role": text, "current_state": "AWAITING_SCRAPE_LOCATION"})
        send_message(self.chat_id, "Location? (e.g., Remote, Nigeria)")

    async def _process_location(self, text):
        self.storage.update(self.chat_id, {"target_location": text, "current_state": "AWAITING_JOB_LINK"})
        msg_id = send_message(self.chat_id, "⏳ Deploying multi-platform scrapers (LinkedIn, RemoteOK, etc.)...")
        
        jobs = await self.scraper.fetch_multi_platform_jobs(self.profile["target_role"], text)
        if not jobs:
            edit_message(self.chat_id, msg_id, "No jobs found.")
            return

        edit_message(self.chat_id, msg_id, "⏳ Analyzing targets against your Master Resume...")
        master = HarvardResume.model_validate(self.profile["master_resume"])
        ranked = self.gemini.rank_jobs(master, jobs)
        
        msg = f"✅ **Top {len(ranked.top_matches)} Curated Matches:**\n\n"
        for job in ranked.top_matches:
            msg += f"🎯 *{job.title}* at {job.company} ({job.platform})\n"
            msg += f"🧠 *Why you fit:* {job.why_fit}\n"
            msg += f"🔗 [Link]({job.link})\n\n"
            
        msg += "*Reply with a specific job link to trigger the Tailor & STARL Interview process.*"
        edit_message(self.chat_id, msg_id, msg)

    # --- Tailoring & STARL Interview Flow ---
    def _ask_tailor(self):
        self.storage.update(self.chat_id, {"current_state": "AWAITING_JOB_DESCRIPTION"})
        send_message(self.chat_id, "Paste the full job description text:")

    async def _process_job_link(self, text):
        if not text.startswith("http"): return
        msg_id = send_message(self.chat_id, "⏳ Extracting job requirements...")
        job_desc = await self.scraper.extract_job_description_async(text)
        self.storage.update(self.chat_id, {"job_desc": job_desc})
        edit_message(self.chat_id, msg_id, "✅ Evaluating gaps based on your history...")
        self._evaluate_and_route(job_desc)

    def _process_direct_job(self, text):
        self.storage.update(self.chat_id, {"job_desc": text})
        send_message(self.chat_id, "Evaluating job gaps...")
        self._evaluate_and_route(text)

    def _evaluate_and_route(self, job_desc):
        master = HarvardResume.model_validate(self.profile["master_resume"])
        gap = self.gemini.gap_interview(master, job_desc)
        
        if gap.needs_interview and gap.questions:
            self.storage.update(self.chat_id, {"current_state": "INTERVIEW_MODE", "questions": gap.questions, "current_q_idx": 0, "qa_responses": ""})
            send_message(self.chat_id, f"To tailor this perfectly using the STARL method, I need some context (Feel free to use **Voice Notes**):\n\n*Q 1/{len(gap.questions)}*: {gap.questions[0]}")
        else:
            self._execute_tailoring(job_desc, "No additions needed.")

    def _process_interview(self, message: dict):
        answer_text = self._extract_text_or_voice(message)
        if not answer_text: return

        idx = self.profile["current_q_idx"]
        questions = self.profile["questions"]
        qa = self.profile.get("qa_responses", "") + f"Q: {questions[idx]}\nA: {answer_text}\n\n"
        
        if idx + 1 < len(questions):
            self.storage.update(self.chat_id, {"current_q_idx": idx + 1, "qa_responses": qa})
            send_message(self.chat_id, f"*Q {idx+2}/{len(questions)}*: {questions[idx+1]}")
        else:
            send_message(self.chat_id, "Got it. Restructuring your experience into a flowing STARL narrative and compiling documents...")
            self._execute_tailoring(self.profile["job_desc"], qa)

    def _execute_tailoring(self, job_desc, qa):
        master = HarvardResume.model_validate(self.profile["master_resume"])
        gh_projects = [ProjectEntry.model_validate(p) for p in self.profile.get("github_projects", [])]
        
        tailored = self.gemini.tailor_resume(master, job_desc, qa, gh_projects)
        cheat_sheet = self.gemini.generate_cheat_sheet(master, job_desc)
        
        pdf_path = "/tmp/Tailored_Resume.pdf"
        export_to_pdf(tailored, pdf_path)
        
        cheat_msg = f"🎯 **Match Score:** {cheat_sheet.match_score}%\n📈 **Strategy:** {cheat_sheet.why_you_win}\n\n📝 **Application Cheat Sheet:**\n"
        for qa_pair in cheat_sheet.likely_form_questions:
            cheat_msg += f"• *Q: {qa_pair.question}*\n  *A:* {qa_pair.recommended_answer}\n\n"
            
        send_doc(self.chat_id, pdf_path, cheat_msg)
        self.storage.update(self.chat_id, {"current_state": "AWAITING_COVER_LETTER_CONFIRM"})
        send_message(self.chat_id, "Do you want a Cover Letter generated? (yes/no)")

    def _process_cover_letter(self, text):
        if text.lower() in ["y", "yes"]:
            master = HarvardResume.model_validate(self.profile["master_resume"])
            letter = self.gemini.cover_letter(master, self.profile["job_desc"])
            send_message(self.chat_id, f"📝 *Cover Letter*\n\n{letter}")
        else:
            send_message(self.chat_id, "Skipped.")
        self.storage.update(self.chat_id, {"current_state": "IDLE"})

@app.function(secrets=[modal.Secret.from_name("resume-agent-secret")], timeout=300)
def process_update_in_background(request_data: dict):
    msg = request_data.get("message")
    if not msg: return
    bot = BotController(msg["chat"]["id"])
    asyncio.run(bot.handle_async(msg))

@app.function(secrets=[modal.Secret.from_name("resume-agent-secret")])
@modal.fastapi_endpoint(method="POST")
def telegram_webhook(request: dict):
    process_update_in_background.spawn(request)
    return {"status": "ok"}

@app.function(schedule=modal.Cron("0 8 * * *", timezone="Africa/Lagos"), secrets=[modal.Secret.from_name("resume-agent-secret")], timeout=600)
def daily_job_scrape_cron():
    from supabase import create_client
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    res = supabase.table("profiles").select("*").eq("cron_enabled", True).execute()
    
    for user in res.data:
        chat_id = user.get("chat_id")
        role, loc = user.get("target_role"), user.get("target_location")
        master_data = user.get("master_resume")
        
        if chat_id and role and loc and master_data:
            scraper = ScraperService()
            gemini = GeminiService()
            
            jobs = asyncio.run(scraper.fetch_multi_platform_jobs(role, loc))
            if jobs:
                master = HarvardResume.model_validate(master_data)
                ranked = gemini.rank_jobs(master, jobs)
                
                msg = f"🌅 **Daily Intel:** Top matches for `{role}` in `{loc}`\n\n"
                for j in ranked.top_matches:
                    msg += f"🔹 *{j.title}* at {j.company} ({j.platform})\n*Why:* {j.why_fit}\n[Apply here]({j.link})\n\n"
                
                msg += "*Reply with a link to trigger the STARL Tailor process. Type /stop to pause alerts.*"
                send_message(chat_id, msg)
                supabase.table("profiles").update({"current_state": "AWAITING_JOB_LINK"}).eq("chat_id", chat_id).execute()