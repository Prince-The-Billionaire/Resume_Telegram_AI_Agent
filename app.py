import os
import json
import io
import logging
import asyncio
import itertools
from typing import Dict, List, Optional
import modal
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ==================== MODAL ENVIRONMENT SETUP ====================
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

# ==================== SCHEMAS ====================
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

class AnalyticsReport(BaseModel):
    ats_score: int
    ats_verdict: str
    ats_rationale: str
    hr_hook_score: int
    hr_feedback: str
    competitive_tier: str
    actionable_improvements: List[str]
    go_no_go_recommendation: str

class GitHubProjectInfo(BaseModel):
    title: str
    description: str
    live_link: Optional[str] = None
    achievements: List[str]

class GitHubAnalysisResult(BaseModel):
    top_projects: List[GitHubProjectInfo]

# ==================== CSS TEMPLATE ====================
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
  ul.bullet-points { list-style-type: disc; margin-left: 1.25rem; margin-top: 0.25rem; }
  ul.bullet-points li { margin-bottom: 0.25rem; font-size: 0.95rem; color: #334155; }
  .skills-container { margin-bottom: 0.5rem; font-size: 0.95rem; }
</style>
"""

# ==================== SERVICES ====================
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
        prompt = f"Parse this raw text resume directly into the structured Harvard schema structure. Reorganize all profile handles into absolute https:// links:\n\n{raw_text}"
        return self._structured(prompt, HarvardResume, 0.1)

    def select_top_github_projects(self, repos_data: List[dict]) -> GitHubAnalysisResult:
        prompt = f"Analyze these GitHub repositories. Extract the top 3 strongest projects based on code complexity and relevance. Create strong bullet point achievements for each.\nRepos:\n{json.dumps(repos_data, indent=2)}"
        return self._structured(prompt, GitHubAnalysisResult, 0.2)

    def gap_interview(self, master: HarvardResume, job_description: str) -> TechnicalGapInterrogator:
        prompt = f"Compare this candidate's profile to the target job description. Identify up to 3 core hard technical components or metrics missing.\nResume:\n{master.model_dump_json()}\nJob Description:\n{job_description}"
        return self._structured(prompt, TechnicalGapInterrogator, 0.2)

    def tailor_resume(self, master: HarvardResume, job_description: str, interview_qa: str, github_projects: List[ProjectEntry]) -> HarvardResume:
        prompt = f"""You are an expert career agent formatting a resume to the Harvard standard.
TAILORING RULES: Do NOT delete existing jobs. Blend answers seamlessly. Add the synthesized GitHub projects into the key_projects section to boost technical density.
Master Profile:\n{master.model_dump_json()}
GitHub Projects to Inject:\n{json.dumps([p.model_dump() for p in github_projects], indent=2)}
Job Description:\n{job_description}\nCandidate's Answers:\n{interview_qa}"""
        return self._structured(prompt, HarvardResume, 0.2)

    def ats_fix(self, tailored: HarvardResume, recommendations: List[str]) -> HarvardResume:
        prompt = f"Revise the tailored resume to directly execute these ATS recommendations.\nResume: {tailored.model_dump_json()}\nRecommendations: {json.dumps(recommendations)}"
        return self._structured(prompt, HarvardResume, 0.2)

    def evaluate(self, tailored: HarvardResume, job_description: str) -> AnalyticsReport:
        prompt = f"You are an elite corporate recruiter. Critique this tailored resume.\nResume: {tailored.model_dump_json()}\nJob: {job_description}"
        return self._structured(prompt, AnalyticsReport, 0.2)

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        prompt = "You are an expert transcriptionist. Transcribe this audio exactly as spoken. Do not summarize or add commentary. Just return the verbatim spoken text."
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                self.types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                prompt
            ]
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
            "last_tailored": {}, "last_recommendations": [], "target_role": "", "target_location": "",
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
                    if r.get("fork"): continue
                    repos_data.append({
                        "name": r.get("name"),
                        "description": r.get("description") or "No description",
                        "homepage": r.get("homepage") or "",
                        "language": r.get("language") or "Code"
                    })
        except Exception as e:
            logger.error(f"GitHub Scraper Error: {e}")
        return repos_data

    async def fetch_job_listings_async(self, role: str, location: str) -> List[Dict]:
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
                    for card in cards[:3]:
                        title = await (await card.query_selector(".base-search-card__title")).inner_text()
                        link = await (await card.query_selector("a.base-card__full-link")).get_attribute("href")
                        results.append({"title": title.strip(), "platform": "LinkedIn", "link": link.split('?')[0]})
                except Exception: pass

                # 2. RemoteOK
                try:
                    await page.goto(f"https://remoteok.com/remote-{role.lower().replace(' ', '-')}-jobs", timeout=10000)
                    jobs = await page.query_selector_all("tr.job")
                    for job in jobs[:2]:
                        title = await (await job.query_selector("h2")).inner_text()
                        link = await job.get_attribute("data-url")
                        results.append({"title": title.strip(), "platform": "RemoteOK", "link": f"https://remoteok.com{link}"})
                except Exception: pass

                # 3. Jobberman
                try:
                    await page.goto(f"https://www.jobberman.com/jobs?q={role.replace(' ', '+')}&l={location.replace(' ', '+')}", timeout=10000)
                    links = await page.query_selector_all("a[href*='/job/']")
                    for a in links[:2]:
                        text = await a.inner_text()
                        href = await a.get_attribute("href")
                        if len(text.strip()) > 5:
                            results.append({"title": text.strip()[:60], "platform": "Jobberman", "link": href})
                except Exception: pass

                await browser.close()
        except Exception as e:
            logger.error(f"Playwright Jobs error: {e}")
        return results

    async def extract_job_description_async(self, url: str) -> str:
        from playwright.async_api import async_playwright
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=20000)
                await page.evaluate("() => document.querySelectorAll('script, style, nav, footer').forEach(el => el.remove())")
                text = await page.inner_text("body")
                await browser.close()
                clean = ' '.join(text.split())
                return clean[:4000] if len(clean) > 100 else f"Fallback Context from: {url}"
        except Exception as e:
            return f"Context extraction failed. Role URL: {url}"

# ==================== TELEGRAM HELPERS ====================
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

def extract_pdf_text(file_id: str) -> str:
    import requests
    from pypdf import PdfReader
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    file_path = tg_api("getFile", {"file_id": file_id})["result"]["file_path"]
    resp = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}")
    reader = PdfReader(io.BytesIO(resp.content))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

def download_tg_voice(file_id: str) -> bytes:
    import requests
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    file_info = tg_api("getFile", {"file_id": file_id})
    file_path = file_info["result"]["file_path"]
    resp = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}")
    return resp.content

def format_job_table(jobs: List[Dict]) -> str:
    if not jobs: return "No jobs found. Try adjusting role/location."
    return "\n".join([f"🔹 *{j['platform']}*: [{j['title']}]({j['link']})" for j in jobs])

def export_to_pdf(data: HarvardResume, output_filename="/tmp/tailored_resume.pdf"):
    from weasyprint import HTML
    skills = "".join([f"<div class='skills-container'><strong>{c.category_name}:</strong> {', '.join(c.subcategories)}</div>" for c in data.technical_skills])
    edu = "".join([f"<div class='item'><div class='item-header'><span class='item-title'>{e.institution}</span><span class='item-date'>{e.duration}</span></div><div class='item-subtitle'>{e.degree}</div></div>" for e in data.education])
    exp = "".join([f"<div class='item'><div class='item-header'><span class='item-title'>{j.company}</span><span class='item-date'>{j.duration}</span></div><div class='item-subtitle'>{j.role}</div><ul class='bullet-points'>{''.join([f'<li>{a}</li>' for a in j.achievements])}</ul></div>" for j in data.work_experience])
    proj = "".join([f"<div class='item'><div class='item-header'><span class='item-title'>{p.title}</span></div><div class='item-subtitle'><a href='{p.link or '#'}'>{p.link or ''}</a></div><ul class='bullet-points'>{''.join([f'<li>{a}</li>' for a in p.achievements])}</ul></div>" for p in data.key_projects])
    
    html = f"""<html><head>{ENHANCED_RESUME_CSS}</head><body>
    <header><h1>{data.personal_info.name}</h1><div class='contact-info'><span>{data.personal_info.email}</span> | <span>{data.personal_info.linkedin}</span> | <span>{data.personal_info.github}</span></div></header>
    <section><h2>Technical Skills</h2>{skills}</section>
    <section><h2>Experience</h2>{exp}</section>
    <section><h2>Key Projects</h2>{proj}</section>
    <section><h2>Education</h2>{edu}</section>
    </body></html>"""
    HTML(string=html).write_pdf(output_filename)

# ==================== BOT CONTROLLER ====================
class BotController:
    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.storage = StorageService()
        self.gemini = GeminiService()
        self.scraper = ScraperService()
        self.profile = self.storage.get_or_create_profile(chat_id)
        self.menu_text = "\n\n📋 *Commands:* /start | /scrape | /tailor | /fixissues | /changeresume"

    async def spinner_task(self, msg_id: int, base_text: str):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        try:
            for frame in itertools.cycle(frames):
                edit_message(self.chat_id, msg_id, f"⏳ {base_text}... `{frame}`")
                await asyncio.sleep(1.2)
        except asyncio.CancelledError:
            pass

    async def handle_async(self, message: dict):
        state = self.profile.get("current_state", "IDLE")
        text = message.get("text", "").strip() if "text" in message else ""

        # Commands override state if they are text
        if text == "/start": return self._start()
        if text in ["/scrape", "/newscrape"]: return self._ask_role()
        if text == "/tailor": return self._ask_tailor()
        if text == "/changeresume": return self._change_resume()
        if text in ["/fixissues", "/fix-issues"]: return self._fix_issues()

        # State Routing
        if state == "AWAITING_MASTER": return self._process_master(message)
        if state == "AWAITING_LINKEDIN": return self._process_linkedin(text)
        if state == "AWAITING_GITHUB": return self._process_github(text)
        if state == "AWAITING_SCRAPE_ROLE": return self._process_role(text)
        if state == "AWAITING_SCRAPE_LOCATION": return await self._process_location(text)
        if state == "AWAITING_JOB_LINK": return await self._process_job_link(text)
        if state == "AWAITING_JOB_DESCRIPTION": return self._process_direct_job(text)
        if state == "INTERVIEW_MODE": return self._process_interview(message)
        if state == "AWAITING_COVER_LETTER_CONFIRM": return self._process_cover_letter(text)

        send_message(self.chat_id, "Unknown command." + self.menu_text)

    def _start(self):
        if self.profile.get("master_resume"):
            send_message(self.chat_id, "Welcome back! Ready to hunt." + self.menu_text)
        else:
            self.storage.update(self.chat_id, {"current_state": "AWAITING_MASTER"})
            send_message(self.chat_id, "Let's begin. Please upload your Master Resume as a PDF.")

    def _process_master(self, message):
        if "document" not in message or not message["document"].get("file_name", "").lower().endswith(".pdf"):
            send_message(self.chat_id, "Please upload a PDF document.")
            return
        send_message(self.chat_id, "Processing PDF...")
        try:
            raw = extract_pdf_text(message["document"]["file_id"])
            parsed = self.gemini.parse_master(raw)
            self.storage.update(self.chat_id, {"master_resume": parsed.model_dump(), "current_state": "AWAITING_LINKEDIN"})
            send_message(self.chat_id, f"Parsed details for {parsed.personal_info.name}. Now, send your LinkedIn URL:")
        except Exception as e:
            send_message(self.chat_id, f"Parse error: {e}")

    def _process_linkedin(self, text):
        self.storage.update(self.chat_id, {"linkedin": text, "current_state": "AWAITING_GITHUB"})
        send_message(self.chat_id, "Got it. Now send your GitHub URL:")

    def _process_github(self, text):
        msg_id = send_message(self.chat_id, "⏳ Deep-analyzing GitHub repositories...")
        self.storage.update(self.chat_id, {"github": text})
        
        raw_repos = self.scraper.scrape_github_repos(text)
        if raw_repos:
            analysis = self.gemini.select_top_github_projects(raw_repos)
            projects = [ProjectEntry(title=p.title, link=p.live_link, achievements=p.achievements).model_dump() for p in analysis.top_projects]
            self.storage.update(self.chat_id, {"github_projects": projects})
            edit_message(self.chat_id, msg_id, "✅ GitHub analyzed! Top projects staged for injection." + self.menu_text)
        else:
            edit_message(self.chat_id, msg_id, "⚠️ No public repos found. Moving on." + self.menu_text)
        self.storage.update(self.chat_id, {"current_state": "IDLE"})

    def _change_resume(self):
        self.storage.update(self.chat_id, {"current_state": "AWAITING_MASTER", "master_resume": {}})
        send_message(self.chat_id, "Upload your new Master Resume PDF.")

    def _ask_role(self):
        self.storage.update(self.chat_id, {"current_state": "AWAITING_SCRAPE_ROLE"})
        send_message(self.chat_id, "What job role are you targeting? (e.g., Full Stack Developer)")

    def _process_role(self, text):
        self.storage.update(self.chat_id, {"target_role": text, "current_state": "AWAITING_SCRAPE_LOCATION"})
        send_message(self.chat_id, "What location? (e.g., Remote, Lagos, New York)")

    async def _process_location(self, text):
        self.storage.update(self.chat_id, {"target_location": text, "current_state": "AWAITING_JOB_LINK"})
        msg_id = send_message(self.chat_id, "⏳ Firing up Playwright...")
        spinner = asyncio.create_task(self.spinner_task(msg_id, "Scraping Jobberman, LinkedIn, & RemoteOK"))
        
        jobs = await self.scraper.fetch_job_listings_async(self.profile["target_role"], text)
        spinner.cancel()
        
        edit_message(self.chat_id, msg_id, f"✅ Jobs Found:\n\n{format_job_table(jobs)}\n\n*Reply with the exact link you want to target.*" + self.menu_text)

    def _ask_tailor(self):
        self.storage.update(self.chat_id, {"current_state": "AWAITING_JOB_DESCRIPTION"})
        send_message(self.chat_id, "Paste the full job description text:")

    async def _process_job_link(self, text):
        if not text.startswith("http"):
            send_message(self.chat_id, "Please send a valid URL.")
            return
            
        msg_id = send_message(self.chat_id, "⏳ Initializing...")
        spinner = asyncio.create_task(self.spinner_task(msg_id, "Extracting JS-rendered job page"))
        
        job_desc = await self.scraper.extract_job_description_async(text)
        spinner.cancel()
        
        self.storage.update(self.chat_id, {"job_desc": job_desc})
        edit_message(self.chat_id, msg_id, "✅ Page extracted. Evaluating gaps...")
        self._evaluate_and_route(job_desc)

    def _process_direct_job(self, text):
        self.storage.update(self.chat_id, {"job_desc": text})
        send_message(self.chat_id, "Evaluating job gaps...")
        self._evaluate_and_route(text)

    def _evaluate_and_route(self, job_desc):
        master = HarvardResume.model_validate(self.profile["master_resume"])
        gap = self.gemini.gap_interview(master, job_desc)
        if gap.needs_interview and gap.questions:
            self.storage.update(self.chat_id, {"current_state": "INTERVIEW_MODE", "questions": gap.questions, "current_q_idx": 0})
            send_message(self.chat_id, f"To tailor perfectly, answer this (Text or Voice under 90s):\n\n*Q 1/{len(gap.questions)}*: {gap.questions[0]}")
        else:
            self._execute_tailoring(job_desc, "No additions needed.")

    def _process_interview(self, message: dict):
        # Handle Voice Note Transcription
        if "voice" in message:
            voice_meta = message["voice"]
            duration = voice_meta.get("duration", 0)
            
            # Duration Check: Reject notes longer than 90 seconds
            if duration > 90:
                send_message(
                    self.chat_id, 
                    f"⚠️ *Voice note too long* ({duration}s).\nPlease keep your answer under 90 seconds, or reply with a text message."
                )
                return

            send_message(self.chat_id, "🎙️ Listening and transcribing...")
            try:
                audio_bytes = download_tg_voice(voice_meta["file_id"])
                answer_text = self.gemini.transcribe_audio(audio_bytes)
                send_message(self.chat_id, f"📝 *Transcript:* {answer_text}")
            except Exception as e:
                send_message(self.chat_id, f"Failed to transcribe audio: {e}. Please type your answer.")
                return
        elif "text" in message:
            answer_text = message["text"].strip()
        else:
            send_message(self.chat_id, "Please reply with text or a voice note.")
            return

        # Proceed with saving the answer
        idx = self.profile["current_q_idx"]
        questions = self.profile["questions"]
        qa = self.profile.get("qa_responses", "") + f"Q: {questions[idx]}\nA: {answer_text}\n\n"
        
        if idx + 1 < len(questions):
            self.storage.update(self.chat_id, {"current_q_idx": idx + 1, "qa_responses": qa})
            send_message(self.chat_id, f"*Q {idx+2}/{len(questions)}*: {questions[idx+1]}\n*(Reply with Text or Voice under 90s)*")
        else:
            send_message(self.chat_id, "Got it. Compiling resume...")
            self._execute_tailoring(self.profile["job_desc"], qa)

    def _execute_tailoring(self, job_desc, qa):
        master = HarvardResume.model_validate(self.profile["master_resume"])
        gh_projects = [ProjectEntry.model_validate(p) for p in self.profile.get("github_projects", [])]
        
        tailored = self.gemini.tailor_resume(master, job_desc, qa, gh_projects)
        report = self.gemini.evaluate(tailored, job_desc)
        
        pdf_path = "/tmp/Tailored_Resume.pdf"
        export_to_pdf(tailored, pdf_path)
        
        send_doc(self.chat_id, pdf_path, f"ATS Score: {report.ats_score}/100\nVerdict: {report.ats_verdict}")
        
        self.storage.update(self.chat_id, {
            "current_state": "AWAITING_COVER_LETTER_CONFIRM", 
            "last_tailored": tailored.model_dump(),
            "last_recommendations": report.actionable_improvements
        })
        send_message(self.chat_id, "Do you want a Cover Letter generated? (yes/no)" + self.menu_text)

    def _process_cover_letter(self, text):
        if text.lower() in ["y", "yes"]:
            master = HarvardResume.model_validate(self.profile["master_resume"])
            letter = self.gemini.cover_letter(master, self.profile["job_desc"])
            send_message(self.chat_id, f"📝 *Cover Letter*\n\n{letter}" + self.menu_text)
        else:
            send_message(self.chat_id, "Skipped." + self.menu_text)
        self.storage.update(self.chat_id, {"current_state": "IDLE"})

    def _fix_issues(self):
        if not self.profile.get("last_tailored"):
            send_message(self.chat_id, "No active tailored resume to fix.")
            return
        send_message(self.chat_id, "Applying HR recommendations...")
        tailored = HarvardResume.model_validate(self.profile["last_tailored"])
        fixed = self.gemini.ats_fix(tailored, self.profile.get("last_recommendations", []))
        
        pdf_path = "/tmp/Fixed_Resume.pdf"
        export_to_pdf(fixed, pdf_path)
        send_doc(self.chat_id, pdf_path, "Fixed ATS Resume based on feedback." + self.menu_text)
        self.storage.update(self.chat_id, {"current_state": "IDLE", "last_tailored": fixed.model_dump()})

# ==================== MODAL BACKGROUND HANDLER ====================
@app.function(secrets=[modal.Secret.from_name("resume-agent-secret")], timeout=300)
def process_update_in_background(request_data: dict):
    msg = request_data.get("message")
    if not msg: return
    chat_id = msg["chat"]["id"]
    bot = BotController(chat_id)
    asyncio.run(bot.handle_async(msg))

# ==================== MODAL WEBHOOK ====================
@app.function(secrets=[modal.Secret.from_name("resume-agent-secret")])
@modal.fastapi_endpoint(method="POST")
def telegram_webhook(request: dict):
    process_update_in_background.spawn(request)
    return {"status": "ok"}

# ==================== CRON JOB ====================
@app.function(schedule=modal.Cron("0 8 * * *", timezone="Africa/Lagos"), secrets=[modal.Secret.from_name("resume-agent-secret")])
def daily_job_scrape_cron():
    from supabase import create_client
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    res = supabase.table("profiles").select("*").execute()
    for user in res.data:
        chat_id, role, loc = user.get("chat_id"), user.get("target_role"), user.get("target_location")
        if chat_id and role and loc:
            scraper = ScraperService()
            jobs = asyncio.run(scraper.fetch_job_listings_async(role, loc))
            msg = f"🌅 Good morning! Fresh jobs for `{role}` in `{loc}`\n\n{format_job_table(jobs)}\n\n*Reply with a link to tailor.*"
            send_message(chat_id, msg)
            supabase.table("profiles").update({"current_state": "AWAITING_JOB_LINK"}).eq("chat_id", chat_id).execute()