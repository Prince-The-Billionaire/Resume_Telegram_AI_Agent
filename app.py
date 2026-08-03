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

app = modal.App("ats-resume-bot", image=image)

# --- Pydantic Models ---
class PersonalInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    linkedin: str
    github: Optional[str] = None          # now optional – can hold GitHub, Behance, or any portfolio URL

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
    achievements: List[str]

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

class GitHubProjectInfo(BaseModel):          # kept name for compatibility; used for any portfolio
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

# --- CSS (Harvard Standard) – UNCHANGED ---
HARVARD_RESUME_CSS = """
<style>
  @page {
      margin: 0.5in;
      size: letter;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { 
      font-family: "Times New Roman", Times, serif; 
      font-size: 12px; 
      color: #000000; 
      line-height: 1.3; 
      background-color: #ffffff; 
      max-width: 100%; 
  }
  header { 
      text-align: center; 
      margin-bottom: 12px; 
  }
  header h1 { 
      font-size: 22px; 
      font-weight: normal; 
      text-transform: uppercase; 
      margin-bottom: 4px;
      letter-spacing: 1px;
  }
  .contact-info { 
      font-size: 12px; 
      text-align: center; 
  }
  .contact-info span { margin: 0 4px; }
  a { 
      color: #000000; 
      text-decoration: none; 
  }
  section { margin-bottom: 12px; }
  section h2 { 
      font-size: 13px; 
      font-weight: bold; 
      text-transform: uppercase; 
      border-bottom: 1px solid #000000; 
      margin-top: 10px; 
      margin-bottom: 6px; 
      padding-bottom: 2px;
  }
  .item { margin-bottom: 8px; }
  .item-header { 
      display: flex; 
      justify-content: space-between; 
      align-items: baseline; 
  }
  .item-title { font-weight: bold; }
  .item-date { font-weight: normal; }
  .item-subtitle { 
      display: flex; 
      justify-content: space-between; 
      align-items: baseline; 
      font-style: italic; 
      margin-bottom: 3px; 
  }
  ul.bullet-points { 
      list-style-type: disc; 
      margin-left: 24px; 
      margin-top: 2px; 
  }
  ul.bullet-points li { 
      margin-bottom: 3px; 
      text-align: justify;
  }
  .skills-container { margin-bottom: 4px; }
  .skills-container strong { font-weight: bold; }
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
        prompt = f"Parse this raw text resume directly into the structured schema. Treat any portfolio/GitHub/Behance links as optional.\n\n{raw_text}"
        return self._structured(prompt, HarvardResume, 0.1)
        
    def generate_master_from_dictation(self, raw_dictation: str) -> HarvardResume:
        prompt = f"""You are an elite career agent converting unstructured dictation into a Harvard-standard Master Resume. 
This candidate may be from ANY field (software, marketing, oil & gas, design, international relations, engineering, etc.).
CRITICAL FORMATTING RULE: For work experience and project 'achievements', use 1 to 3 punchy bullet points using the STARL method (Situation, Task, Action, Result, Learning). Do not use long paragraphs.
If the candidate has no technical skills, still produce a clean skills section with relevant professional skills.
Dictation:\n{raw_dictation}"""
        return self._structured(prompt, HarvardResume, 0.3)

    def select_top_github_projects(self, repos_data: List[dict]) -> GitHubAnalysisResult:
        prompt = f"""Analyze these portfolio / repository items. Extract the top 3 strongest projects based on impact and complexity. 
Create strong, bulleted achievements for each using STARL logic. Works for code, design, marketing campaigns, engineering projects, etc.
Items:\n{json.dumps(repos_data, indent=2)}"""
        return self._structured(prompt, GitHubAnalysisResult, 0.2)

    def select_job_specific_github_projects(self, repos_data: List[dict], job_description: str) -> GitHubAnalysisResult:
        prompt = f"""Analyze these portfolio / repository items and the target Job Description. 
Extract the top 2 to 3 most relevant projects that specifically demonstrate the skills required. 
Create strong, bulleted achievements for each using STARL logic. Field-agnostic (tech, marketing, design, engineering, etc.).
Job Description:\n{job_description}\n\nItems:\n{json.dumps(repos_data, indent=2)}"""
        return self._structured(prompt, GitHubAnalysisResult, 0.2)

    def gap_interview(self, master: HarvardResume, job_description: str) -> TechnicalGapInterrogator:
        prompt = f"""Compare the candidate's profile to the job description. Identify missing components (technical or professional).
If a gap exists, frame your question by referencing their PAST EXPERIENCE to prompt a STARL response. 
This works for any industry. Do not assume the candidate is a software engineer.
Resume:\n{master.model_dump_json()}\nJob Description:\n{job_description}"""
        return self._structured(prompt, TechnicalGapInterrogator, 0.3)

    def tailor_resume(self, master: HarvardResume, job_description: str, interview_qa: str, github_projects: List[ProjectEntry]) -> HarvardResume:
        prompt = f"""You are an expert career agent formatting a Harvard-standard resume for ANY profession.
TAILORING RULES: 
1. Rewrite the work experience 'achievements' into 1 to 3 high-impact bullet points per role strictly using the STARL method.
2. Intelligently weave critical missing keywords and technologies/skills from the Job Description into the candidate's past roles based on their dictated answers. Ensure maximum ATS compatibility.
3. Replace the existing key_projects section entirely with the newly provided Projects (they may come from GitHub, Behance, or user dictation).
Master Profile:\n{master.model_dump_json()}
Job-Specific Projects to inject:\n{json.dumps([p.model_dump() for p in github_projects], indent=2)}
Job Description:\n{job_description}\nCandidate's Dictated Answers:\n{interview_qa}"""
        return self._structured(prompt, HarvardResume, 0.2)

    def rank_jobs(self, master: HarvardResume, scraped_jobs: List[Dict]) -> JobRankerResult:
        prompt = f"""Analyze these scraped jobs against the master resume.
Return ONLY the top 3-5 jobs where the candidate has the highest probability of passing ATS based on their specific background.
Provide a 1-sentence 'why_fit' explanation. Field-agnostic.
Resume:\n{master.model_dump_json()}\nScraped Jobs:\n{json.dumps(scraped_jobs)}"""
        return self._structured(prompt, JobRankerResult, 0.3)

    def generate_cheat_sheet(self, master: HarvardResume, job_description: str) -> ApplicationCheatSheet:
        prompt = f"""Analyze this job description and resume. 
1. Score the match. 2. Explain why they win. 3. Predict the 3 most difficult custom application form questions and generate exact copy-paste answers.
Works for any industry.
Master Resume:\n{master.model_dump_json()}\nJob:\n{job_description}"""
        return self._structured(prompt, ApplicationCheatSheet, 0.3)

    def cover_letter(self, master: HarvardResume, job_description: str) -> str:
        prompt = f"Write a highly tailored, compelling cover letter based on this resume and job description. Ready to send without placeholder brackets. Suitable for any profession.\n\nResume:\n{master.model_dump_json()}\n\nJob:\n{job_description}"
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
        api_url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=15"
        repos_data = []
        try:
            resp = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if resp.status_code == 200:
                for r in resp.json():
                    if not r.get("fork"):
                        repos_data.append({
                            "name": r.get("name"),
                            "description": r.get("description"),
                            "language": r.get("language"),
                            "url": r.get("html_url")
                        })
        except Exception:
            pass
        return repos_data

    async def scrape_behance_projects(self, behance_url: str) -> List[dict]:
        """Lightweight Behance project extraction via Playwright."""
        from playwright.async_api import async_playwright
        projects = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(behance_url, timeout=20000)
                # Behance project cards
                cards = await page.query_selector_all("div.ProjectCoverNeue-root-orQ, div.Cover-cover-s5i, a[href*='/gallery/']")
                for card in cards[:12]:
                    try:
                        title_el = await card.query_selector("h3, .ProjectCoverNeue-title-*, .title")
                        title = (await title_el.inner_text()).strip() if title_el else "Untitled Project"
                        link = await card.get_attribute("href") or ""
                        if link and not link.startswith("http"):
                            link = "https://www.behance.net" + link
                        projects.append({"name": title, "description": "", "url": link})
                    except Exception:
                        continue
                await browser.close()
        except Exception as e:
            logger.error(f"Behance scrape error: {e}")
        return projects

    def scrape_portfolio(self, url: str) -> List[dict]:
        """Unified entry point – detects domain and routes."""
        if not url:
            return []
        url_l = url.lower()
        if "github.com" in url_l:
            return self.scrape_github_repos(url)
        # Behance is async, so we handle it in the caller when needed
        return []

    async def scrape_portfolio_async(self, url: str) -> List[dict]:
        if not url:
            return []
        url_l = url.lower()
        if "github.com" in url_l:
            return self.scrape_github_repos(url)
        if "behance.net" in url_l:
            return await self.scrape_behance_projects(url)
        return []

    async def fetch_multi_platform_jobs(self, role: str, location: str) -> List[Dict]:
        from playwright.async_api import async_playwright
        results = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                try:
                    await page.goto(f"https://www.linkedin.com/jobs/search/?keywords={role.replace(' ', '%20')}&location={location.replace(' ', '%20')}", timeout=15000)
                    cards = await page.query_selector_all("div.base-search-card")
                    for card in cards[:10]:
                        title = await (await card.query_selector(".base-search-card__title")).inner_text()
                        company = await (await card.query_selector(".base-search-card__subtitle")).inner_text()
                        link = await (await card.query_selector("a.base-card__full-link")).get_attribute("href")
                        results.append({"title": title.strip(), "company": company.strip(), "platform": "LinkedIn", "link": link.split('?')[0]})
                except Exception:
                    pass
                
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

def set_bot_commands():
    commands = [
        {"command": "start", "description": "Start the bot & see main menu"},
        {"command": "generate", "description": "Dictate your Master Resume (Voice/Text)"},
        {"command": "scrape", "description": "Search jobs on LinkedIn & more"},
        {"command": "tailor", "description": "Tailor resume using STARL method"},
        {"command": "github", "description": "Sync GitHub / Behance / Portfolio projects"},
        {"command": "changeresume", "description": "Upload a new Master Resume PDF"},
        {"command": "stop", "description": "Pause daily job alerts"}
    ]
    tg_api("setMyCommands", {"commands": commands})

def send_message(chat_id: int, text: str) -> int:
    res = tg_api("sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    return res.get("result", {}).get("message_id")

def edit_message(chat_id: int, msg_id: int, text: str):
    tg_api("editMessageText", {"chat_id": chat_id, "message_id": msg_id, "text": text, "parse_mode": "Markdown"})

def send_doc(chat_id: int, file_path: str, caption: str = ""):
    with open(file_path, "rb") as f:
        # Keep caption short – Telegram hard limit is 1024 chars
        safe_caption = (caption or "")[:1000]
        tg_api("sendDocument", {"chat_id": chat_id, "caption": safe_caption}, files={"document": f})

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
    
    # Technical Skills
    skills = "".join([f"<div class='skills-container'><strong>{c.category_name}:</strong> {', '.join(c.subcategories)}</div>" for c in data.technical_skills])
    
    # Education
    edu = "".join([
        f"<div class='item'>"
        f"<div class='item-header'><span class='item-title'>{e.institution}</span><span class='item-date'>{e.duration}</span></div>"
        f"<div class='item-subtitle'><span>{e.degree}</span><span>{e.grade or ''}</span></div>"
        f"</div>" for e in data.education
    ])
    
    # Experience
    exp = "".join([
        f"<div class='item'>"
        f"<div class='item-header'><span class='item-title'>{j.company}</span><span class='item-date'>{j.duration}</span></div>"
        f"<div class='item-subtitle'><span>{j.role}</span></div>"
        f"<ul class='bullet-points'>{''.join([f'<li>{ach}</li>' for ach in j.achievements])}</ul>"
        f"</div>" for j in data.work_experience
    ])
    
    # Projects
    proj = "".join([
        f"<div class='item'>"
        f"<div class='item-header'><span class='item-title'>{p.title}</span><span class='item-date'><a href='{p.link or '#'}'>{p.link or ''}</a></span></div>"
        f"<ul class='bullet-points'>{''.join([f'<li>{a}</li>' for a in p.achievements])}</ul>"
        f"</div>" for p in data.key_projects
    ])

    # Interests
    interests = "".join([f"<li><strong>{i.label}:</strong> {i.details}</li>" for i in data.interests]) if hasattr(data, 'interests') and data.interests else ""
    interests_section = f"<section><h2>Interests</h2><ul class='bullet-points'>{interests}</ul></section>" if interests else ""
    
    # Contact line – handle missing / non-GitHub portfolio gracefully
    pi = data.personal_info
    contact_parts = []
    if pi.linkedin:
        contact_parts.append(f"<span>LinkedIn: <a href='{pi.linkedin}'>{pi.linkedin.split('/')[-1] if '/' in pi.linkedin else pi.linkedin}</a></span>")
    if pi.email:
        contact_parts.append(f"<span>Email: <a href='mailto:{pi.email}'>{pi.email}</a></span>")
    if pi.github:
        label = "GitHub" if "github.com" in pi.github.lower() else "Portfolio"
        display = pi.github.split('/')[-1] if '/' in pi.github else pi.github
        contact_parts.append(f"<span>{label}: <a href='{pi.github}'>{display}</a></span>")
    if pi.phone:
        contact_parts.append(f"<span>Phone: {pi.phone}</span>")
    
    contact_html = " | ".join(contact_parts)
    
    html = f"""<html><head>{HARVARD_RESUME_CSS}</head><body>
    <header>
        <h1>{pi.name}</h1>
        <div class='contact-info'>{contact_html}</div>
    </header>
    <section><h2>Technical Skills</h2>{skills}</section>
    <section><h2>Education</h2>{edu}</section>
    <section><h2>Work Experience</h2>{exp}</section>
    <section><h2>Key Projects</h2>{proj}</section>
    {interests_section}
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

    def _refresh_profile(self):
        self.profile = self.storage.get_or_create_profile(self.chat_id)
        return self.profile

    def _get_master_resume(self):
        self._refresh_profile()
        master_data = self.profile.get("master_resume") or {}
        if not master_data or "personal_info" not in master_data:
            send_message(self.chat_id, "⚠️ Please create or upload your Master Resume first using /generate or /changeresume.")
            return None
        try:
            return HarvardResume.model_validate(master_data)
        except Exception:
            send_message(self.chat_id, "⚠️ Your stored Master Resume is invalid. Please regenerate it with /generate.")
            return None

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
        if state.startswith("GENERATE_"): return self._process_generation(state, message)

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

        send_message(self.chat_id, "Command recognized. Tap the 'Menu' button or type /start to see available options.")

    # --- Core Commands ---
    def _start(self):
        set_bot_commands()
        self.storage.update(self.chat_id, {"cron_enabled": True})
        
        welcome_msg = """🤖 **Welcome to your AI Career Agent!** 
        
I work for *any* profession – software, marketing, oil & gas, design, international relations, engineering, and more.

Tap the **Menu button** or use:

🎙️ **/generate** – Build your Master Resume from scratch (voice or text).
🔍 **/scrape** – Tell me your target role + location. I scrape & rank jobs.
✂️ **/tailor** – Paste a job description. I rewrite your resume with the **STARL** method and inject the most relevant projects.
🐙 **/github** – Sync GitHub, Behance, or any portfolio and update your Master Resume.
📄 **/changeresume** – Upload a new Master Resume PDF.
🛑 **/stop** – Pause daily job alerts.

Upload a PDF as your Master Resume, or type /generate to start dictating."""

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
            self._refresh_profile()
            send_message(self.chat_id, "Parsed. Now send your LinkedIn URL:")
        except Exception as e:
            send_message(self.chat_id, f"Parse error: {e}")

    def _process_linkedin(self, text):
        self.storage.update(self.chat_id, {"linkedin": text, "current_state": "AWAITING_GITHUB"})
        send_message(self.chat_id, "Received. Now send your **GitHub, Behance, or any portfolio URL**.\n\nIf you don’t have one, just type `skip`.")

    def _process_github_initial(self, text):
        url = text.strip()
        if url.lower() == "skip":
            url = ""
        self.storage.update(self.chat_id, {"github": url, "current_state": "IDLE"})
        if url:
            # fire-and-forget scan
            asyncio.create_task(self._execute_portfolio_scan(url))
        else:
            send_message(self.chat_id, "✅ Profile saved. You can always add projects later with /github or during /tailor.")

    # --- Portfolio / GitHub Sync Flow ---
    def _trigger_github_sync(self):
        current = self.profile.get("github", "")
        if current:
            asyncio.create_task(self._execute_portfolio_scan(current))
        else:
            self.storage.update(self.chat_id, {"current_state": "AWAITING_GITHUB_SYNC"})
            send_message(self.chat_id, "No portfolio linked yet. Send your GitHub, Behance, or other portfolio URL (or type `skip`):")

    def _process_github_sync(self, text: str):
        url = text.strip()
        if url.lower() == "skip":
            url = ""
        self.storage.update(self.chat_id, {"github": url, "current_state": "IDLE"})
        if url:
            asyncio.create_task(self._execute_portfolio_scan(url))
        else:
            send_message(self.chat_id, "Okay, no portfolio linked.")

    async def _execute_portfolio_scan(self, portfolio_url: str):
        msg_id = send_message(self.chat_id, "⏳ Deep-scanning your portfolio for projects...")
        raw_items = await self.scraper.scrape_portfolio_async(portfolio_url)
        
        if raw_items:
            analysis = self.gemini.select_top_github_projects(raw_items)
            projects = [ProjectEntry(title=p.title, link=p.live_link or p.description, achievements=p.achievements).model_dump() for p in analysis.top_projects]
            self.storage.update(self.chat_id, {"github_projects": projects})
            
            master_data = self.profile.get("master_resume")
            if master_data:
                master_data["key_projects"] = projects
                # also keep the portfolio URL on the personal_info
                if "personal_info" in master_data:
                    master_data["personal_info"]["github"] = portfolio_url
                self.storage.update(self.chat_id, {"master_resume": master_data})
                
                master = HarvardResume.model_validate(master_data)
                pdf_path = "/tmp/Updated_Master.pdf"
                export_to_pdf(master, pdf_path)
                
                edit_message(self.chat_id, msg_id, "✅ Portfolio synced! Generating your updated Master Resume...")
                send_doc(self.chat_id, pdf_path, "📄 Updated Master Resume with your latest projects.")
            else:
                edit_message(self.chat_id, msg_id, "✅ Portfolio synced! Projects staged for your next application.")
        else:
            edit_message(self.chat_id, msg_id, "⚠️ No public projects found (or the site could not be scraped). You can still dictate projects during /generate or /tailor.")

    # --- Generation Flow ---
    def _start_generate(self):
        self.storage.update(self.chat_id, {"current_state": "GENERATE_PERSONAL", "generate_buffer": ""})
        send_message(self.chat_id, "Let's build your Master Resume. You can type or send **Voice Notes**.\n\nFirst, tell me your full name, email, phone (optional), LinkedIn, and any portfolio URL (GitHub / Behance / other – or say “none”).")

    def _process_generation(self, state: str, message: dict):
        text_input = self._extract_text_or_voice(message)
        if not text_input: return
        
        buffer = self.profile.get("generate_buffer", "") + f"\n[{state}]: {text_input}"
        
        if state == "GENERATE_PERSONAL":
            self.storage.update(self.chat_id, {"current_state": "GENERATE_EDU", "generate_buffer": buffer})
            send_message(self.chat_id, "Got it. Now dictate your Education (University, Degree, Graduation Date).")
        elif state == "GENERATE_EDU":
            self.storage.update(self.chat_id, {"current_state": "GENERATE_EXP", "generate_buffer": buffer})
            send_message(self.chat_id, "Great. Now list your Work Experience. Tell me the company, role, duration, and what you achieved.")
        elif state == "GENERATE_EXP":
            self.storage.update(self.chat_id, {"current_state": "GENERATE_SKILLS", "generate_buffer": buffer})
            send_message(self.chat_id, "Almost done. Dictate your Skills (technical, professional, tools, languages – whatever is relevant to your field).")
        elif state == "GENERATE_SKILLS":
            self.storage.update(self.chat_id, {"current_state": "GENERATE_PROJECTS", "generate_buffer": buffer})
            send_message(self.chat_id, "Final step – tell me about your key projects, campaigns, case studies, or notable work (title + what you achieved). If you have none, just say “none”.")
        elif state == "GENERATE_PROJECTS":
            self.storage.update(self.chat_id, {"current_state": "IDLE", "generate_buffer": buffer})
            msg_id = send_message(self.chat_id, "⏳ Compiling and formatting your Master Resume...")
            master = self.gemini.generate_master_from_dictation(buffer)
            self.storage.update(self.chat_id, {"master_resume": master.model_dump()})
            self._refresh_profile()
            pdf_path = "/tmp/Generated_Master.pdf"
            export_to_pdf(master, pdf_path)
            send_doc(self.chat_id, pdf_path, "✅ Master Resume Generated and saved!")
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
        send_message(self.chat_id, "Target Role? (e.g., Python Developer, Marketing Manager, Reservoir Engineer)")

    def _process_role(self, text):
        self.storage.update(self.chat_id, {"target_role": text, "current_state": "AWAITING_SCRAPE_LOCATION"})
        send_message(self.chat_id, "Location? (e.g., Remote, Nigeria, Lagos)")

    async def _process_location(self, text):
        self.storage.update(self.chat_id, {"target_location": text, "current_state": "AWAITING_JOB_LINK"})
        msg_id = send_message(self.chat_id, "⏳ Deploying multi-platform scrapers...")
        
        jobs = await self.scraper.fetch_multi_platform_jobs(self.profile["target_role"], text)
        if not jobs:
            edit_message(self.chat_id, msg_id, "No jobs found right now.")
            return

        edit_message(self.chat_id, msg_id, "⏳ Analyzing targets against your Master Resume...")
        master = self._get_master_resume()
        if master is None:
            return
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
        master = self._get_master_resume()
        if master is None:
            return
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
        send_message(self.chat_id, "⏳ Looking for the most relevant projects from your portfolio...")
        
        portfolio_url = self.profile.get("github") or ""
        gh_projects = []
        
        if portfolio_url:
            # synchronous path for GitHub; async path already handled earlier if needed
            raw_repos = self.scraper.scrape_portfolio(portfolio_url)
            if raw_repos:
                analysis = self.gemini.select_job_specific_github_projects(raw_repos, job_desc)
                gh_projects = [ProjectEntry(title=p.title, link=p.live_link, achievements=p.achievements) for p in analysis.top_projects]
        
        # Fallback to previously stored projects
        if not gh_projects:
            gh_projects = [ProjectEntry.model_validate(p) for p in self.profile.get("github_projects", [])]
        
        send_message(self.chat_id, "⏳ Formatting tailored PDF...")
        master = self._get_master_resume()
        if master is None:
            return
        tailored = self.gemini.tailor_resume(master, job_desc, qa, gh_projects)
        cheat_sheet = self.gemini.generate_cheat_sheet(master, job_desc)
        
        pdf_path = "/tmp/Tailored_Resume.pdf"
        export_to_pdf(tailored, pdf_path)
        
        # CRITICAL FIX: short caption only → PDF always arrives
        send_doc(self.chat_id, pdf_path, "📄 Your tailored Harvard-style resume is ready!")
        
        # Cheat sheet as a separate message (no length limit issues)
        cheat_msg = f"🎯 **Match Score:** {cheat_sheet.match_score}%\n📈 **Strategy:** {cheat_sheet.why_you_win}\n\n📝 **Application Cheat Sheet:**\n"
        for qa_pair in cheat_sheet.likely_form_questions:
            cheat_msg += f"• *Q: {qa_pair.question}*\n  *A:* {qa_pair.recommended_answer}\n\n"
        send_message(self.chat_id, cheat_msg)
        
        self.storage.update(self.chat_id, {"current_state": "AWAITING_COVER_LETTER_CONFIRM"})
        send_message(self.chat_id, "Do you want a Cover Letter generated? (yes/no)")

    def _process_cover_letter(self, text):
        if text.lower() in ["y", "yes"]:
            master = self._get_master_resume()
        if master is None:
            return
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