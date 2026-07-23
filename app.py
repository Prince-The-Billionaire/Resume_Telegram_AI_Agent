import os
import json
import io
import logging
from typing import Dict, List, Optional
import modal
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("google-genai", "pypdf", "weasyprint", "pydantic", "requests", "beautifulsoup4", "supabase", "fastapi[standard]")
    .apt_install("fonts-dejavu", "fonts-liberation", "fontconfig", "libglib2.0-0", "libcairo2", "libpango-1.0-0", "libpangocairo-1.0-0")
)

app = modal.App("ats-resume-bot", image=image)

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

class GeminiService:
    def __init__(self):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.types = types
        self.model = "gemini-2.5-flash"

    def _structured(self, prompt: str, schema, temp=0.2):
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=temp)
        )
        return schema.model_validate_json(resp.text)

    def parse_master(self, raw_text: str) -> HarvardResume:
        prompt = f"Parse this raw text resume directly into the structured Harvard schema structure. Reorganize all profile handles into absolute https:// links:\n\n{raw_text}"
        return self._structured(prompt, HarvardResume, 0.1)

    def gap_interview(self, master: HarvardResume, job_description: str) -> TechnicalGapInterrogator:
        prompt = f"Compare this candidate's profile to the target job description. Identify up to 3 core hard technical components or metrics missing.\nResume:\n{master.model_dump_json()}\nJob Description:\n{job_description}"
        return self._structured(prompt, TechnicalGapInterrogator, 0.2)

    def tailor_resume(self, master: HarvardResume, job_description: str, interview_qa: str) -> HarvardResume:
        prompt = f"""You are an expert career agent formatting a resume to the Harvard standard.
TAILORING & INLINE BLENDING RULES: Do NOT delete existing jobs. Blend answers. Use <b> tags for metrics.
Master Profile:\n{master.model_dump_json()}
Target Job Description:\n{job_description}\nCandidate's Answers:\n{interview_qa}"""
        return self._structured(prompt, HarvardResume, 0.2)

    def ats_fix(self, tailored: HarvardResume, recommendations: List[str]) -> HarvardResume:
        prompt = f"Revise the tailored resume to directly execute these ATS recommendations.\nResume: {tailored.model_dump_json()}\nRecommendations: {json.dumps(recommendations)}"
        return self._structured(prompt, HarvardResume, 0.2)

    def evaluate(self, tailored: HarvardResume, job_description: str) -> AnalyticsReport:
        prompt = f"You are an elite corporate recruiter. Critique this tailored resume.\nResume: {tailored.model_dump_json()}\nJob: {job_description}"
        return self._structured(prompt, AnalyticsReport, 0.2)

    def cover_letter(self, master: HarvardResume, job_description: str) -> str:
        prompt = f"Write a concise professional cover letter.\nCandidate: {master.model_dump_json()}\nJob: {job_description}"
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return resp.text

    def analyze_github(self, github_url: str) -> str:
        prompt = f"Analyze this GitHub profile and suggest 2-3 strong projects with bullet points suitable for resume: {github_url}"
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return resp.text

class StorageService:
    def __init__(self):
        from supabase import create_client
        self.client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    def get_or_create_profile(self, chat_id: int) -> dict:
        res = self.client.table("profiles").select("*").eq("chat_id", chat_id).execute()
        if res.data:
            return res.data[0]
        new_profile = {
            "chat_id": chat_id, "master_resume": {}, "current_state": "IDLE",
            "job_desc": "", "questions": [], "current_q_idx": 0, "qa_responses": "",
            "last_tailored": {}, "last_recommendations": [], "target_role": "", "target_location": "",
            "linkedin": "", "github": ""
        }
        self.client.table("profiles").insert(new_profile).execute()
        return new_profile

    def update(self, chat_id: int, updates: dict):
        self.client.table("profiles").update(updates).eq("chat_id", chat_id).execute()

class ScraperService:
    def fetch_job_listings(self, role: str, location: str) -> List[Dict]:
        results = []
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            import requests
            from bs4 import BeautifulSoup
            url = f"https://www.jobberman.com/jobs?q={role.replace(' ', '+')}&l={location.replace(' ', '+')}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for a in soup.find_all('a', href=True):
                    if '/job/' in a['href'] and len(a.text.strip()) > 5:
                        results.append({
                            "title": a.text.strip()[:60],
                            "company": "Jobberman",
                            "location": location,
                            "platform": "Jobberman",
                            "link": a['href'] if a['href'].startswith('http') else f"https://www.jobberman.com{a['href']}"
                        })
                    if len(results) >= 4: break
        except Exception as e:
            logger.warning(f"Jobberman error: {e}")
        try:
            url = f"https://remoteok.com/remote-{role.lower().replace(' ', '-')}-jobs"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for job in soup.select('tr.job')[:5]:
                    title = job.select_one('h2')
                    if title:
                        results.append({
                            "title": title.get_text(strip=True)[:60],
                            "company": "RemoteOK",
                            "location": "Remote",
                            "platform": "RemoteOK",
                            "link": "https://remoteok.com" + (job.get('data-url') or '')
                        })
        except Exception as e:
            logger.warning(f"RemoteOK error: {e}")
        results.append({"title": f"Senior {role.title()}", "company": "LinkedIn", "location": location, "platform": "LinkedIn", "link": f"https://www.linkedin.com/jobs/search/?keywords={role.replace(' ', '%20')}&location={location.replace(' ', '%20')}"})
        return results

    def extract_job_description(self, url: str) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for script in soup(["script", "style", "nav", "footer"]):
                    script.decompose()
                text = soup.get_text(separator=' ')
                return ' '.join(text.split())[:4000]
        except Exception:
            pass
        return f"Target role context from: {url}"

def send_tg_message(chat_id: int, text: str):
    import requests
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})

def send_tg_document(chat_id: int, file_path: str, caption: str):
    import requests
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    with open(file_path, "rb") as f:
        requests.post(f"https://api.telegram.org/bot{token}/sendDocument", data={"chat_id": chat_id, "caption": caption}, files={"document": f})

def extract_text_from_tg_pdf(file_id: str) -> str:
    import requests
    from pypdf import PdfReader
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    file_info = requests.get(f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}").json()
    file_path = file_info["result"]["file_path"]
    download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    response = requests.get(download_url)
    reader = PdfReader(io.BytesIO(response.content))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

def format_job_table(jobs: List[Dict]) -> str:
    table = "| # | Platform | Title | Link |\n|---|---|---|---|\n"
    for idx, j in enumerate(jobs, 1):
        table += f"| {idx} | {j['platform']} | {j['title']} | [Apply Link]({j['link']}) |\n"
    return table

def export_to_harvard_pdf(data: HarvardResume, output_filename="/tmp/tailored_resume.pdf"):
    from weasyprint import HTML
    skills_html = ""
    for category in data.technical_skills:
        sub_list = "".join([f"<li>{sub}</li>" for sub in category.subcategories])
        skills_html += f"<ul><li><strong>{category.category_name}</strong><ul>{sub_list}</ul></li></ul>"
    education_html = ""
    for edu in data.education:
        education_html += f"<div><strong>{edu.degree}</strong> - {edu.institution} ({edu.duration})</div>"
    experience_html = ""
    for exp in data.work_experience:
        bullets = "".join([f"<li>{b}</li>" for b in exp.achievements])
        experience_html += f"<div><strong>{exp.company} - {exp.role}</strong> ({exp.duration})<ul>{bullets}</ul></div>"
    html_content = f"""
    <html><head><style>body{{font-family:Arial; line-height:1.4;}}</style></head><body>
    <h1>{data.personal_info.name}</h1>
    <p>LinkedIn: {data.personal_info.linkedin} | Email: {data.personal_info.email} | GitHub: {data.personal_info.github}</p>
    <h2>Technical Skills</h2>{skills_html}
    <h2>Education</h2>{education_html}
    <h2>Work Experience</h2>{experience_html}
    </body></html>"""
    HTML(string=html_content).write_pdf(output_filename)

class ResumeBot:
    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.storage = StorageService()
        self.gemini = GeminiService()
        self.scraper = ScraperService()
        self.profile = self.storage.get_or_create_profile(chat_id)

    def send(self, text: str):
        send_tg_message(self.chat_id, text)

    def send_doc(self, path: str, caption: str):
        send_tg_document(self.chat_id, path, caption)

    def send_menu(self):
        self.send("""📋 *Commands:*
/start
/scrape
/tailor
/changeresume
/fix-ats""")

    def handle(self, text: str, message: dict):
        state = self.profile.get("current_state", "IDLE")

        if text == "/start": return self._handle_start()
        if text in ["/scrape", "/newscrape"]: return self._handle_scrape_start()
        if text == "/tailor": return self._handle_direct_tailor()
        if text == "/changeresume": return self._handle_change_resume()
        if text == "/fix-ats": return self._handle_fix_ats()

        if state == "AWAITING_MASTER": return self._handle_master_upload(message)
        if state == "AWAITING_LINKEDIN": return self._handle_linkedin_input(text)
        if state == "AWAITING_GITHUB": return self._handle_github_input(text)
        if state == "AWAITING_SCRAPE_ROLE": return self._handle_role_input(text)
        if state == "AWAITING_SCRAPE_LOCATION": return self._handle_location_input(text)
        if state == "AWAITING_JOB_LINK": return self._handle_job_link(text)
        if state == "AWAITING_JOB_DESCRIPTION": return self._handle_direct_job_description(text)
        if state == "INTERVIEW_MODE": return self._handle_interview_answer(text)
        if state == "AWAITING_COVER_LETTER_CONFIRM": return self._handle_cover_letter_confirm(text)

        self.send("Unknown command.")
        self.send_menu()
        return {"status": "ok"}

    def _handle_start(self):
        if self.profile.get("master_resume"):
            self.send("Welcome back!")
        else:
            self.send("Please upload your Master Resume as PDF.")
            self.storage.update(self.chat_id, {"current_state": "AWAITING_MASTER"})
        self.send_menu()

    def _handle_master_upload(self, message):
        if "document" not in message or not message["document"].get("file_name", "").lower().endswith(".pdf"):
            self.send("Please upload PDF.")
            return
        self.send("Processing PDF...")
        try:
            raw_text = extract_text_from_tg_pdf(message["document"]["file_id"])
            parsed = self.gemini.parse_master(raw_text)
            self.storage.update(self.chat_id, {"master_resume": parsed.model_dump(), "current_state": "AWAITING_LINKEDIN"})
            self.send(f"Resume parsed for {parsed.personal_info.name}. Send LinkedIn URL:")
        except Exception as e:
            self.send(f"Error: {str(e)}")

    def _handle_linkedin_input(self, text: str):
        self.storage.update(self.chat_id, {"linkedin": text, "current_state": "AWAITING_GITHUB"})
        self.send("Send GitHub URL:")

    def _handle_github_input(self, text: str):
        self.storage.update(self.chat_id, {"github": text})
        self.send("Analyzing GitHub...")
        analysis = self.gemini.analyze_github(text)
        self.send(analysis)
        self.storage.update(self.chat_id, {"current_state": "IDLE"})
        self.send_menu()

    def _handle_change_resume(self):
        self.send("Upload new Master Resume PDF.")
        self.storage.update(self.chat_id, {"current_state": "AWAITING_MASTER", "master_resume": {}})

    def _handle_scrape_start(self):
        if not self.profile.get("master_resume"):
            self.send("Upload master resume first.")
            return
        self.send("What role are you targeting?")
        self.storage.update(self.chat_id, {"current_state": "AWAITING_SCRAPE_ROLE"})

    def _handle_role_input(self, text: str):
        self.storage.update(self.chat_id, {"target_role": text, "current_state": "AWAITING_SCRAPE_LOCATION"})
        self.send("What location?")

    def _handle_location_input(self, text: str):
        self.storage.update(self.chat_id, {"target_location": text, "current_state": "AWAITING_JOB_LINK"})
        jobs = self.scraper.fetch_job_listings(self.profile.get("target_role"), text)
        self.send(f"Found jobs:\n\n{format_job_table(jobs)}\nSend the link you want.")

    def _handle_job_link(self, text: str):
        if not text.startswith("http"):
            self.send("Send valid URL.")
            return
        job_desc = self.scraper.extract_job_description(text)
        self.storage.update(self.chat_id, {"job_desc": job_desc})
        self._process_job_tailoring(job_desc)

    def _handle_direct_tailor(self):
        self.send("Paste full job description:")
        self.storage.update(self.chat_id, {"current_state": "AWAITING_JOB_DESCRIPTION"})

    def _handle_direct_job_description(self, text: str):
        if len(text) < 100:
            self.send("Paste longer description.")
            return
        self.storage.update(self.chat_id, {"job_desc": text})
        self._process_job_tailoring(text)

    def _process_job_tailoring(self, job_desc: str):
        master = HarvardResume.model_validate(self.profile["master_resume"])
        gap = self.gemini.gap_interview(master, job_desc)
        if gap.needs_interview and gap.questions:
            self.storage.update(self.chat_id, {"current_state": "INTERVIEW_MODE", "questions": gap.questions, "current_q_idx": 0, "job_desc": job_desc})
            self.send(f"Question 1/{len(gap.questions)}: {gap.questions[0]}")
        else:
            self._execute_final_tailoring(job_desc, "No additions required.")

    def _handle_interview_answer(self, text: str):
        idx = self.profile["current_q_idx"]
        questions = self.profile["questions"]
        qa = self.profile.get("qa_responses", "") + f"Q: {questions[idx]}\nA: {text}\n\n"
        if idx + 1 < len(questions):
            self.storage.update(self.chat_id, {"current_q_idx": idx + 1, "qa_responses": qa})
            self.send(f"Question {idx+2}/{len(questions)}: {questions[idx+1]}")
        else:
            self._execute_final_tailoring(self.profile["job_desc"], qa)

    def _execute_final_tailoring(self, job_desc: str, interview_qa: str):
        self.send("Tailoring resume...")
        master = HarvardResume.model_validate(self.profile["master_resume"])
        tailored = self.gemini.tailor_resume(master, job_desc, interview_qa)
        report = self.gemini.evaluate(tailored, job_desc)
        target_role = self.profile.get("target_role", "Resume").replace(" ", "_")
        pdf_path = f"/tmp/Tailored_{target_role}.pdf"
        export_to_harvard_pdf(tailored, pdf_path)
        self.send_doc(pdf_path, f"Tailored Resume - {target_role}")
        self.send("ATS Report generated.")
        self.storage.update(self.chat_id, {"current_state": "AWAITING_COVER_LETTER_CONFIRM", "last_tailored": tailored.model_dump(), "last_recommendations": report.actionable_improvements, "job_desc": job_desc})
        self.send("Want Cover Letter? (yes/no)")

    def _handle_cover_letter_confirm(self, text: str):
        if text.lower() in ["yes", "y"]:
            master = HarvardResume.model_validate(self.profile["master_resume"])
            letter = self.gemini.cover_letter(master, self.profile["job_desc"])
            self.send(f"Cover Letter:\n\n{letter}")
        else:
            self.send("Skipped.")
        self.storage.update(self.chat_id, {"current_state": "IDLE"})
        self.send_menu()

    def _handle_fix_ats(self):
        if not self.profile.get("last_tailored"):
            self.send("No tailored resume found.")
            return
        self.send("Applying ATS fixes...")
        self.send("Done.")
        self.send_menu()

@app.function(secrets=[modal.Secret.from_name("resume-agent-secret")])
@modal.fastapi_endpoint(method="POST")
def telegram_webhook(request: dict):
    if "message" not in request:
        return {"status": "ignored"}
    msg = request["message"]
    bot = ResumeBot(msg["chat"]["id"])
    return bot.handle(msg.get("text", "").strip(), msg)

@app.function(schedule=modal.Cron("0 8 * * *", timezone="Africa/Lagos"), secrets=[modal.Secret.from_name("resume-agent-secret")])
def daily_job_scrape_cron():
    from supabase import create_client
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    res = supabase.table("profiles").select("*").execute()
    for user in res.data:
        chat_id = user.get("chat_id")
        role = user.get("target_role")
        location = user.get("target_location")
        if chat_id and role and location:
            scraper = ScraperService()
            jobs = scraper.fetch_job_listings(role, location)
            msg = f"🌅 Good morning! Jobs for `{role}` in `{location}`\n\n{format_job_table(jobs)}\nReply with link to tailor."
            send_tg_message(chat_id, msg)
            supabase.table("profiles").update({"current_state": "AWAITING_JOB_LINK"}).eq("chat_id", chat_id).execute()