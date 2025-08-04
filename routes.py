"""
app/routes.py  –  aligned with templates
"""

from __future__ import annotations

import json
import logging
import os
import sys
import html
from enum import Enum
from dataclasses import dataclass
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, List, Dict

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
    session,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename

# ── import app.* even when executed from repo root ───────────────────────────
sys.path.append(str(Path(__file__).parent.parent))

from app.auth import auth_manager, get_current_user
from config import LANG_NAMES, PRIMARY_LANGUAGES, SECONDARY_LANGUAGES, TARGET_LANGUAGES, LANGUAGE_NORMALIZATION



bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)

# ───────────────────────────── helper(s) ──────────────────────────────────────
def sse_format(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"

# Add this after your imports (around line 35)
def sanitize_for_log(value: Any, max_length: int = 100) -> str:
    """Sanitize any value for safe logging."""
    if value is None:
        return "None"
    
    # Convert to string
    str_value = str(value)
    
    # Remove newlines and carriage returns
    str_value = str_value.replace('\n', ' ').replace('\r', ' ')
    
    # Limit length
    if len(str_value) > max_length:
        str_value = str_value[:max_length] + "..."
    
    return str_value

class ErrorHandlingMode(Enum):
    """Error handling strategies"""
    STRICT = "strict"
    PERMISSIVE = "permissive"
    INTERACTIVE = "interactive"

class FormatLevel(Enum):
    """Formatting preservation levels"""
    MINIMAL = "minimal"
    BASIC = "basic"
    FULL = "full"

@dataclass
class FormattingIssue:
    """Represents a formatting issue found in the text"""
    line_number: int
    issue_type: str
    description: str
    original_text: str
    suggested_fix: Optional[str] = None

@dataclass
class ProcessingResult:
    """Result of text processing"""
    success: bool
    html_content: Optional[str] = None
    issues: List[FormattingIssue] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None

class TextFormatter:
    """Handles text formatting detection and conversion"""
    
    PATTERNS = {
        'bullet_list': re.compile(r'^[\s]*[-*•]\s+(.+)$', re.MULTILINE),
        'numbered_list': re.compile(r'^[\s]*(\d+)[.)]\s+(.+)$', re.MULTILINE),
        'blockquote': re.compile(r'^>\s*(.+)$', re.MULTILINE),
        'code_fence': re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL),
        'inline_code': re.compile(r'`([^`]+)`'),
        'bold': re.compile(r'\*\*([^*]+)\*\*'),
        'italic': re.compile(r'\*([^*]+)\*'),
        'heading': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
        'horizontal_rule': re.compile(r'^[\s]*[-*_]{3,}[\s]*$', re.MULTILINE),
        'excessive_whitespace': re.compile(r'[ \t]{3,}'),
        'multiple_blank_lines': re.compile(r'\n{3,}'),
        'long_line': re.compile(r'^.{300,}$', re.MULTILINE),
    }
    
    @staticmethod
    def normalize_whitespace(text: str, preserve_formatting: bool = True) -> str:
        """Normalize whitespace while preserving intentional formatting"""
        if text.startswith('\ufeff'):
            text = text[1:]
        
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        if not preserve_formatting:
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
        else:
            text = re.sub(r'[ \t]{4,}', '    ', text)
            text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        return text.strip()

class ComprehensiveHTMLWrapper:
    """Main class for wrapping text as HTML with comprehensive formatting"""
    
    def __init__(self, 
                 error_mode: ErrorHandlingMode = ErrorHandlingMode.PERMISSIVE,
                 format_level: FormatLevel = FormatLevel.FULL):
        self.error_mode = error_mode
        self.format_level = format_level
        self.formatter = TextFormatter()
    
    def wrap_text_as_html(self, text_content: str, primary_lang: str = "en") -> ProcessingResult:
        """Convert text to HTML with formatting preservation"""
        if not text_content or not text_content.strip():
            return ProcessingResult(
                success=False,
                issues=[FormattingIssue(0, "empty_input", "No text content provided", "")],
                warnings=["Empty input received"]
            )
            issues = []
        warnings = []
        metadata = {
            'original_length': len(text_content),
            'detected_formats': []
        }
        
        try:
            # Normalize text
            normalized_text = self.formatter.normalize_whitespace(
                text_content, 
                preserve_formatting=(self.format_level != FormatLevel.MINIMAL)
            )
            
            # Process based on format level
            if self.format_level == FormatLevel.MINIMAL:
                html_body = self._process_minimal(normalized_text)
            else:
                html_body = self._process_full(normalized_text, metadata)
            
            # Build final HTML
            html_content = self._build_html_document(html_body, primary_lang, metadata)
            
            return ProcessingResult(
                success=True,
                html_content=html_content,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error wrapping text: {e}")
            return ProcessingResult(
                success=False,
                issues=[FormattingIssue(0, "processing_error", str(e), text_content[:100])]
            )
    
    def _process_minimal(self, text: str) -> str:
        """Minimal processing - just paragraphs and line breaks"""
        text = html.escape(text)
        paragraphs = text.split('\n\n')
        html_parts = []
        
        for para in paragraphs:
            para = para.strip()
            if para:
                para_html = para.replace('\n', '<br>\n')
                html_parts.append(f'    <p>{para_html}</p>')
        
        return '\n'.join(html_parts)
    
    def _process_full(self, text: str, metadata: Dict) -> str:
        """Full processing with all formatting features"""
        # First escape HTML entities
        text = html.escape(text)
        
        # Process formatting
        html_parts = []
        blocks = text.split('\n\n')
        
        for block in blocks:
            if not block.strip():
                continue
        # Apply formatting patterns
            block = self._apply_inline_formatting(block, metadata)
            html_parts.append(f'    <p>{block}</p>')
        
        return '\n'.join(html_parts)
    
    def _apply_inline_formatting(self, text: str, metadata: Dict) -> str:
        """Apply inline formatting like bold, italic, code"""
        # Bold
        if '**' in text:
            text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
            metadata['detected_formats'].append('bold')
        
        # Italic
        if '*' in text:
            text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
            metadata['detected_formats'].append('italic')
        
        # Inline code
        if '`' in text:
            text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
            metadata['detected_formats'].append('inline_code')
        
        # Line breaks
        text = text.replace('\n', '<br>\n')
        
        return text
    
    def _build_html_document(self, body_content: str, lang: str, metadata: Dict) -> str:
        """Build the final HTML document"""
        return f"""<!DOCTYPE html>
<html lang="{html.escape(lang)}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
        }}
        .text-content {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        p {{
            margin: 0 0 1em 0;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
  <div class="text-content">
{body_content}
  </div>
</body>
</html>"""

# HTML Detection Functions
def detect_html_content(text: str) -> tuple[bool, float]:
    """Detect if text content is actually HTML"""
    if not text or len(text.strip()) < 10:
        return False, 0.0
    
    text_lower = text.strip().lower()
    score = 0.0
    
    # Check for DOCTYPE or html tag
    if text_lower.startswith('<!doctype') or text_lower.startswith('<html'):
        score += 0.5
    
    # Check for common HTML tags
    html_tags = ['<head', '<body', '<title>', '<div', '<p>', '<h1', '<meta']
    tag_count = sum(1 for tag in html_tags if tag in text_lower)
    score += min(tag_count * 0.1, 0.4)
    
    # Check for closing tags
    if '</html>' in text_lower or '</body>' in text_lower:
        score += 0.1
    
    is_html = score >= 0.5
    return is_html, score


def detect_sql_content(text: str) -> tuple[bool, float]:
    """Detect if text content is actually SQL"""
    if not text or len(text.strip()) < 10:
        return False, 0.0
    
    text_upper = text.upper()
    score = 0.0
    
    # Major SQL statement keywords (strong indicators)
    major_sql_keywords = [
        'CREATE TABLE', 'INSERT INTO', 'UPDATE', 'DELETE FROM', 
        'SELECT', 'ALTER TABLE', 'DROP TABLE', 'CREATE DATABASE'
    ]
    
    for keyword in major_sql_keywords:
        if keyword in text_upper:
            score += 0.3
    
    # Common SQL keywords
    sql_keywords = [
        'WHERE', 'FROM', 'JOIN', 'VALUES', 'SET', 'PRIMARY KEY', 
        'FOREIGN KEY', 'CONSTRAINT', 'INDEX', 'TRIGGER', 'PROCEDURE'
    ]
    
    keyword_count = sum(1 for keyword in sql_keywords if keyword in text_upper)
    score += min(keyword_count * 0.1, 0.4)
    
    # Check for SQL patterns
    sql_patterns = [
        r';\s*$',  # Statements ending with semicolon
        r'\(\s*[^)]+\s*\)',  # Parentheses (common in SQL)
        r'--\s+\w+',  # SQL comments
        r'/\*.*?\*/',  # Multi-line comments
        r'\bVARCHAR\s*\(\d+\)',  # Data type definitions
        r'\bINT\b|\bTEXT\b|\bDATE\b',  # Common data types
    ]
    
    import re
    pattern_matches = sum(1 for pattern in sql_patterns 
                         if re.search(pattern, text, re.IGNORECASE | re.DOTALL))
    score += min(pattern_matches * 0.05, 0.2)
    
    # Penalty for HTML-like content
    if '<' in text and '>' in text:
        score -= 0.3
    
    is_sql = score >= 0.5
    return is_sql, min(score, 1.0)

def detect_python_content(text: str) -> tuple[bool, float]:
    """Detect if text content is actually Python code"""
    if not text or len(text.strip()) < 10:
        return False, 0.0
    
    text_stripped = text.strip()
    score = 0.0
    
    # Check for shebang
    if text_stripped.startswith('#!') and 'python' in text_stripped.split('\n')[0]:
        score += 0.4
    
    # Major Python keywords (strong indicators)
    major_python_keywords = [
        'import ', 'from ', 'def ', 'class ', 'if __name__', 
        'async def', 'await ', 'try:', 'except:', 'finally:'
    ]
    
    for keyword in major_python_keywords:
        if keyword in text:
            score += 0.2
    
    # Python-specific patterns
    python_patterns = [
        r'^import\s+\w+',  # Import statements
        r'^from\s+\w+\s+import',  # From imports
        r'def\s+\w+\s*\(',  # Function definitions
        r'class\s+\w+[\s\(:]',  # Class definitions
        r':\s*$',  # Lines ending with colon
        r'^\s{4,}',  # Indentation
        r'if\s+__name__\s*==\s*["\']__main__["\']',  # Main guard
        r'@\w+',  # Decorators
        r'self\.',  # Self references
        r'print\s*\(',  # Print function
    ]
    
    pattern_matches = sum(1 for pattern in python_patterns 
                         if re.search(pattern, text, re.MULTILINE))
    score += min(pattern_matches * 0.05, 0.3)
    
    # Django/Jinja2 template indicators
    template_patterns = [
        r'{%\s*\w+', r'%}',  # Django/Jinja2 tags
        r'{{\s*\w+', r'}}',  # Template variables
        r'{%\s*trans\s*%}', r'{%\s*blocktrans\s*%}',  # Translation tags
    ]
    
    template_matches = sum(1 for pattern in template_patterns 
                          if re.search(pattern, text))
    if template_matches >= 2:
        score += 0.2
    
    # Penalty for HTML-like content (unless it's a template)
    if '<html' in text.lower() and template_matches < 2:
        score -= 0.3
    
    # Check for Python built-ins usage
    builtins = ['print', 'range', 'len', 'str', 'int', 'float', 'list', 'dict', 'set']
    builtin_count = sum(1 for builtin in builtins if f'{builtin}(' in text)
    score += min(builtin_count * 0.02, 0.1)
    
    is_python = score >= 0.4
    return is_python, min(score, 1.0)


def smart_process_text_input(text_content: str, primary_lang: str, session_path: Path) -> Dict:
    """Intelligently process text input based on content detection"""
    
    # Detect content type
    is_html, html_confidence = detect_html_content(text_content)
    is_sql, sql_confidence = detect_sql_content(text_content)
    is_python, python_confidence = detect_python_content(text_content)
    
    uploads_dir = session_path / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    
    # Determine file type based on confidence scores
    confidences = [
        ('python', python_confidence, is_python),
        ('sql', sql_confidence, is_sql),
        ('html', html_confidence, is_html)
    ]
    confidences.sort(key=lambda x: x[1], reverse=True)
    best_match = confidences[0]
    if best_match[2] and best_match[1] >= 0.4:
        file_type, confidence = best_match[0], best_match[1]
        if file_type == 'python':
            
            python_filename = "pasted_python_input.py"
            python_path = uploads_dir / python_filename
            with open(python_path, 'w', encoding='utf-8') as f:
              f.write(text_content)
            return {
                'success': True,
                'processed_as': 'python',
                'filename': python_filename,
                'is_python': True,
                'confidence': confidence,
                'file_type': 'python'
            }
        elif file_type == 'sql':
            
            sql_filename = "pasted_sql_input.sql"
            sql_path = uploads_dir / sql_filename
            with open(sql_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            return {
                'success': True,
                'processed_as': 'sql',
                'filename': sql_filename,
                'is_sql': True,
                'confidence': confidence,
                'file_type': 'sql'
            }   
    
    elif is_html:
        
        
        # Save as HTML directly
        html_filename = "pasted_html_input.html"
        html_path = uploads_dir / html_filename
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        return {
            'success': True,
            'processed_as': 'html',
            'filename': html_filename,
            'is_html': True,
            'confidence': html_confidence,
            'file_type': 'html'
        }
    else:
        logger.info("Processing as plain text with formatting")
        wrapper = ComprehensiveHTMLWrapper(error_mode=ErrorHandlingMode.PERMISSIVE, format_level=FormatLevel.FULL)
        result = wrapper.wrap_text_as_html(text_content, primary_lang)
        if result.success:
            html_filename = "text_input.html"
            html_path = uploads_dir / html_filename
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(result.html_content)
            return {
                'success': True,
                'processed_as': 'text',
                'filename': html_filename,
                'is_html': False,
                'detected_formats': result.metadata.get('detected_formats', []),
                'warnings': result.warnings
            }
        else:
            return {
                'success': False,
                'error': 'formatting_failed',
                'issues': result.issues
            }
        
            
        

# NEW: Simplified wrapper function that replaces the old wrap_text_as_html
def wrap_text_as_html(text_content: str, primary_lang: str = "en") -> str:
    """
    Simple wrapper function for backward compatibility.
    This replaces the old wrap_text_as_html function.
    """
    wrapper = ComprehensiveHTMLWrapper(
        error_mode=ErrorHandlingMode.PERMISSIVE,
        format_level=FormatLevel.FULL
    )
    
    result = wrapper.wrap_text_as_html(text_content, primary_lang)
    
    if result.success:
        return result.html_content
    else:
        # Fallback to minimal HTML
        escaped = html.escape(text_content)
        return f"""<!DOCTYPE html>
<html lang="{primary_lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <div class="text-content">
    <p>{escaped.replace(chr(10), '<br>' + chr(10))}</p>
  </div>
</body>
</html>"""

def save_session_metadata(session_path: Path, input_type: str, 
                         original_text: str = None, extra_metadata: Dict = None) -> None:
    """Save session metadata to track input type and original content."""
    metadata = {
        "input_type": input_type,
        "created_at": time.time(),
    }
    
    if input_type == "text" and original_text:
        metadata["original_text"] = original_text
        metadata["word_count"] = len(original_text.split())
        metadata["char_count"] = len(original_text)
    
    # Add any extra metadata
    if extra_metadata:
        metadata.update(extra_metadata)
    
    metadata_file = session_path / "session_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def get_session_metadata(session_path: Path) -> dict:
    """Get session metadata, returns empty dict if not found."""
    metadata_file = session_path / "session_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read session metadata: {e}")
            
    return {}

# ─────────────────── background pipeline wrapper ─────────────────────────────
def run_pipeline_async(
    app,
    session_id: str,
    html_files: list[str],
    primary_lang: str,
    secondary_lang: Optional[str],
    target_lang: str,
    enable_refinement: bool = True,
    refinement_mode: str = "none",
) -> None:
    """Run the heavy pipeline in a background thread."""
    import sys
    from app.services.progress_tracker import ProgressTracker
    from app.services.pipeline_runner import PipelineRunner

    # Simply get logger - no handler configuration needed
    log = logging.getLogger("pipeline_async")
    
    log.info("Starting pipeline for session %s", sanitize_for_log(session_id))

    with app.app_context():
        tracker = getattr(app, "progress_tracker", ProgressTracker())
        tracker.update(session_id, 1, "Starting content extraction…")

        # 🔥 FIXED: Proper session path resolution
        try:
            if hasattr(app, "session_manager") and app.session_manager:
                session_path = app.session_manager.get_session_path(session_id)
                log.info("Using session manager path: %s", session_path)
            else:
                # If session manager fails, recreate it
                log.warning("Session manager not available, recreating...")
                from app.services.session_manager import SessionManager
                app.session_manager = SessionManager()
                session_path = app.session_manager.get_session_path(session_id)
                log.info("Created new session manager, path: %s", sanitize_for_log(session_path))
                
        except Exception as e:
            # Last resort: Use project-relative path instead of /tmp
            log.error("Session manager failed: %s", e)
            project_root = Path(__file__).parent.parent  # Go up to project root
            session_path = project_root / "sessions" / session_id
            session_path.mkdir(parents=True, exist_ok=True)
            

        # Log the final path for debugging
        log.info("Final session path: %s", sanitize_for_log(session_path))
        log.info("Session path exists: %s", session_path.exists())

        runner = PipelineRunner(session_path, os.environ.copy())
        try:
            ok = runner.run_batch(
                html_filenames=html_files,
                primary_lang=primary_lang,
                secondary_lang=secondary_lang,
                target_lang=target_lang,
                enable_refinement=enable_refinement,
                  refinement_mode=refinement_mode,)
            log.info(
                "Pipeline finished %s for %s",
                "OK" if ok else "with issues",
                session_id,
            )
        except Exception as exc:  # pragma: no cover
            log.exception("Pipeline failed for %s", sanitize_for_log(session_id))
            tracker.fail(sanitize_for_log(session_id), "pipeline_error", str(exc))
# ───────────────────────────── UI pages ───────────────────────────────────────
# ───────────────────────────── LOGIN ROUTES ────────────────────────────────────
@bp.route("/login", methods=["GET", "POST"])
def login():
    """Simple login page"""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if auth_manager.validate_user(username, password):
            session["authenticated"] = True
            session["username"] = username
            session.permanent = True
            
            logger.info(f"User {sanitize_for_log(username)} logged in successfully")
            return redirect(url_for("main.index"))
        else:
            
            return render_template("login.html", error="Invalid username or password")
    
    return render_template("login.html")

@bp.route("/guest-login", methods=["POST"])
def guest_login():
    """Create a guest session and redirect to main app"""
    auth_manager.create_guest_session()
    logger.info("Guest user accessed the application")
    return redirect(url_for("main.index"))


@bp.route("/logout")
def logout():
    """Simple logout"""
    username = session.get("username", "Unknown")
    session.clear()
    logger.info(f"User {sanitize_for_log(username)} logged out")
    return redirect(url_for("main.login"))


@bp.route("/")
def index() -> str:
    if not session.get("authenticated"):
        return redirect(url_for("main.login"))
    
    # Build language dictionaries from config
    primary_langs = {code: LANG_NAMES[code] for code in PRIMARY_LANGUAGES}
    secondary_langs = {"": "None"}  # Start with None option
    secondary_langs.update({code: LANG_NAMES[code] for code in SECONDARY_LANGUAGES if code})  # Skip empty string
    
    # For target languages, convert from uppercase to lowercase and get names
    target_langs = {}
    for code in TARGET_LANGUAGES:
        lower_code = LANGUAGE_NORMALIZATION[code]  # Convert ZH -> zh, etc.
        target_langs[code] = LANG_NAMES[lower_code]
    
    return render_template(
        "index.html",
        primary_langs=primary_langs,
        secondary_langs=secondary_langs, 
        target_langs=target_langs,
        constants={"MAX_FILE_SIZE": 50 * 1024 * 1024},
        current_user=get_current_user(),
    )

@bp.route("/processing/<session_id>")
def processing(session_id: str) -> str:
    if hasattr(current_app, "session_manager"):
        current_app.session_manager.get_session_path(session_id)  # 404s if bad
    
    # Determine the correct results URL based on session type
    if hasattr(current_app, "session_manager"):
        try:
            spath = current_app.session_manager.get_session_path(session_id)
            metadata = get_session_metadata(spath)
            results_url = "/textresults/" + session_id if metadata.get("input_type") == "text" else "/results/" + session_id
        except Exception:
            results_url = "/results/" + session_id  # Default fallback
    else:
        results_url = "/results/" + session_id
    
    return render_template(
        "processing.html",
        session_id=session_id,
        sse_url=f"/api/progress-stream/{session_id}",
        results_url=results_url,
        current_user=get_current_user(),
    )

@bp.route("/results/<session_id>")
def results(session_id: str) -> str:
    """Render the results page with a list of ZIPs for this session."""
    
    # 🔄 CHECK SESSION TYPE AND REDIRECT TEXT SESSIONS
    if hasattr(current_app, "session_manager"):
        try:
            spath = current_app.session_manager.get_session_path(session_id)
            metadata = get_session_metadata(spath)
            
            # AUTO-REDIRECT: Text sessions → textresults.html
            if metadata.get("input_type") == "text":
                return redirect(url_for("main.textresults", session_id=session_id))
        except Exception as e:
            logger.warning(f"Could not check session metadata for {sanitize_for_log(session_id)}: {e}")
            
    
    # Continue with normal HTML file session logic
    zip_files: list[str] = []
    if hasattr(current_app, "session_manager"):
        spath = current_app.session_manager.get_session_path(session_id)
        rdir = spath / "results"
        if rdir.exists():
            zip_files = sorted(f.name for f in rdir.glob("*.zip"))

    return render_template(
        "results.html",
        session_id=session_id,
        zip_files=zip_files,
        current_user=get_current_user(),
    )
    
@bp.route("/textresults/<session_id>")
def textresults(session_id: str) -> str:
    """Render the text results page showing original text and translations."""
    if not session.get("authenticated"):
        return redirect(url_for("main.login"))
    
    original_text = ""
    deepl_blocks = {}
    openai_blocks = {}
    error_message = None
    zip_files = []
    
    if hasattr(current_app, "session_manager"):
        try:
            spath = current_app.session_manager.get_session_path(session_id)
            
            # Get session metadata
            metadata = get_session_metadata(spath)
            if metadata.get("input_type") != "text":
                # Redirect non-text sessions to regular results
                return redirect(url_for("main.results", session_id=session_id))
            
            # Get original text
            original_text = metadata.get("original_text", "")
            
            # Find the text file (should be only one)
            uploads_dir = spath / "uploads"
            html_files = list(uploads_dir.glob("*.html")) if uploads_dir.exists() else []
            
            if html_files:
                # Use the first (and should be only) HTML file
                html_file = html_files[0]
                basename = html_file.stem
                
                # Load DeepL translations
                deepl_file = spath / "translated" / basename / "segments_only.json"
                if deepl_file.exists():
                    with open(deepl_file, 'r', encoding='utf-8') as f:
                        deepl_data = json.load(f)
                        if isinstance(deepl_data, dict):
                            deepl_blocks = deepl_data
                
                # Load OpenAI translations
                openai_file = spath / "refined" / basename / "openai_translations.json"
                if openai_file.exists():
                    with open(openai_file, 'r', encoding='utf-8') as f:
                        openai_data = json.load(f)
                        if isinstance(openai_data, dict):
                            openai_blocks = openai_data
            
            # Get available ZIP files
            rdir = spath / "results"
            if rdir.exists():
                zip_files = sorted(f.name for f in rdir.glob("*.zip"))
                
        except Exception as e:
            logger.error(f"Error loading text results for {sanitize_for_log(session_id)}: {e}")
            error_message = f"Error loading results: {str(e)}"
    
    return render_template(
        "textresults.html",
        session_id=session_id,
        original_text=original_text,
        deepl_blocks=deepl_blocks,
        openai_blocks=openai_blocks,
        zip_files=zip_files,
        error_message=error_message,
        current_user=get_current_user(),
    )
# Replace your metrics route in routes.py with this corrected version:

@bp.route("/metrics/<session_id>")
def metrics(session_id: str) -> str:
    """Render the metrics page with analytics data for the session."""
    if not session.get("authenticated"):
        return redirect(url_for("main.login"))
    
    # Initialize data containers
    consolidated_data = None
    memory_data = None
    
    if hasattr(current_app, "session_manager"):
        try:
            spath = current_app.session_manager.get_session_path(session_id)
            
            # Try to load consolidated metrics
            consolidated_file = spath / "refined" / "consolidated_metrics.json"
            if consolidated_file.exists():
                with open(consolidated_file, 'r', encoding='utf-8') as f:
                    consolidated_data = json.load(f)
            
            # FIXED: Correct path for memory usage data
            memory_file = spath / "translated" / "memory_usage_batch.json"
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading metrics data for session {sanitize_for_log(session_id)}: {e}")           
    
    return render_template(
        "metrics.html",
        session_id=session_id,
        consolidated_data=consolidated_data,
        memory_data=memory_data,
        current_user=get_current_user(),
    )

# Add these new routes to your existing routes.py file
# Insert after your existing download routes (around line 280-300)

@bp.route("/download-json/<session_id>/consolidated-metrics")
def download_consolidated_metrics(session_id: str):
    """Download consolidated_metrics.json with original filename"""
    if not session.get("authenticated"):
        return redirect(url_for("main.login"))
    
    if hasattr(current_app, "session_manager"):
        try:
            spath = current_app.session_manager.get_session_path(session_id)
            metrics_file = spath / "refined" / "consolidated_metrics.json"
            
            if metrics_file.exists():
                return send_from_directory(
                    metrics_file.parent, 
                    metrics_file.name, 
                    as_attachment=True
                )
            else:
                return jsonify(error="Consolidated metrics file not found"), 404
                
        except Exception as e:
            
            return jsonify(error="Download failed"), 500
    
    return jsonify(error="Session not found"), 404

@bp.route("/download-json/<session_id>/memory-batch")
def download_memory_batch(session_id: str):
    """Download memory_usage_batch.json with original filename"""
    if not session.get("authenticated"):
        return redirect(url_for("main.login"))
    
    if hasattr(current_app, "session_manager"):
        try:
            spath = current_app.session_manager.get_session_path(session_id)
            memory_file = spath / "translated" / "memory_usage_batch.json"
            
            if memory_file.exists():
                return send_from_directory(
                    memory_file.parent, 
                    memory_file.name, 
                    as_attachment=True
                )
            else:
                return jsonify(error="Memory batch file not found"), 404
                
        except Exception as e:
            
            return jsonify(error="Download failed"), 500
    
    return jsonify(error="Session not found"), 404


@bp.route("/api/available-files/<session_id>")
def api_available_files(session_id: str):
    """Get available files for translation comparison"""
    if not session.get("authenticated"):
        return jsonify(error="Authentication required"), 401
    
    available_files = []
    
    if hasattr(current_app, "session_manager"):
        try:
            spath = current_app.session_manager.get_session_path(session_id)
            extracted_dir = spath / "extracted"
            
            if extracted_dir.exists():
                for file_dir in extracted_dir.iterdir():
                    if file_dir.is_dir():
                        sentences_file = file_dir / "translatable_flat_sentences.json"
                        if sentences_file.exists():
                            available_files.append(file_dir.name)
            
            return jsonify(success=True, files=sorted(available_files))
                            
        except Exception as e:
            logger.error(f"Error loading files for session {sanitize_for_log(session_id)}: {e}")
            return jsonify(success=False, error=str(e)), 500
    
    return jsonify(success=False, error="Session manager not available"), 500

@bp.route("/language/<session_id>/<filename>")
def language_comparison(session_id: str, filename: str) -> str:
    """Show block comparison page for a specific file."""
    if not session.get("authenticated"):
        return redirect(url_for("main.login"))
    
    # Sanitize filename to prevent directory traversal
    filename = secure_filename(filename)
    
    original_blocks = {}
    deepl_blocks = {}
    openai_blocks = {}
    error_message = None
    
    if hasattr(current_app, "session_manager"):
        try:
            spath = current_app.session_manager.get_session_path(session_id)
            
            # Verify the session belongs to the authenticated user
            if not spath.exists():
                return jsonify(error="Session not found"), 404
            
            # Load original text (with segment format, will be merged in frontend)
            sentences_file = spath / "extracted" / filename / "translatable_flat_sentences.json"
            if sentences_file.exists():
                with open(sentences_file, 'r', encoding='utf-8') as f:
                    sentences_data = json.load(f)
                    
                # Flatten all categories and keep segment format
                for category in sentences_data.values():
                    if isinstance(category, list):  # Ensure it's a list
                        for block_info in category:
                            if isinstance(block_info, dict):  # Ensure it's a dict
                                for block_id, text in block_info.items():
                                    if block_id != "tag":  # Skip tag fields
                                        # Handle special case where block IDs are combined
                                        if "=" in block_id:
                                            # Split and add each ID separately
                                            ids = block_id.split("=")
                                            for single_id in ids:
                                                single_id = single_id.strip()
                                                if single_id:  # Ensure not empty
                                                    original_blocks[single_id] = text
                                        else:
                                            original_blocks[block_id] = text
            
            # Load DeepL translations (with segment format, will be merged in frontend)
            deepl_file = spath / "translated" / filename / "segments_only.json"
            if deepl_file.exists():
                with open(deepl_file, 'r', encoding='utf-8') as f:
                    deepl_data = json.load(f)
                    # Ensure it's a dictionary before assigning
                    if isinstance(deepl_data, dict):
                        deepl_blocks = deepl_data
            
            # Load OpenAI translations (with segment format, will be merged in frontend)
            openai_file = spath / "refined" / filename / "openai_translations.json"
            if openai_file.exists():
                with open(openai_file, 'r', encoding='utf-8') as f:
                    openai_data = json.load(f)
                    # Ensure it's a dictionary before assigning
                    if isinstance(openai_data, dict):
                        openai_blocks = openai_data
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {sanitize_for_log(session_id)}/{sanitize_for_log(filename)}: {e}")
            error_message = f"Error reading JSON data: Invalid format"
        except Exception as e:
            logger.error(f"Error loading translation data for {sanitize_for_log(session_id)}/{sanitize_for_log(filename)}: {e}")
            error_message = f"Error loading translation data: {str(e)}"
    else:
        error_message = "Session manager not available"
    
    return render_template(
        "language_comparison.html",
        session_id=session_id,
        filename=filename,
        original_blocks=original_blocks,
        deepl_blocks=deepl_blocks,
        openai_blocks=openai_blocks,
        error_message=error_message,
        current_user=get_current_user(),
    )

# ─────────────────────────────── API: upload ────────────────────────────────
@bp.route("/api/upload", methods=["POST"])
def api_upload():
    # Get form data
    input_type = request.form.get("input_type", "file")
    primary = request.form.get("primary_lang", "en")
    target = request.form.get("target_lang", "fr").upper()
    secondary = request.form.get("secondary_lang") or None
    refinement_mode = request.form.get("refinement_mode", "none")
    enable_refinement = refinement_mode != "none"
   
    
    if not hasattr(current_app, "session_manager") or not current_app.session_manager:
        return jsonify(error="server_error", message="Session manager missing"), 500

    try:
        spath = current_app.session_manager.create_session()
    except Exception as exc:
        logger.exception("Cannot create session dir")
        return jsonify(error="server_error", message=str(exc)), 500

    sid = spath.name
    upload_dir = spath / "uploads"
    upload_dir.mkdir(exist_ok=True)

    saved: list[str] = []

    if input_type == "text":
        # Handle text input with smart detection
        text_content = request.form.get("text_content", "").strip()
        
        if not text_content:
            return jsonify(error="invalid_input", message="No text content provided"), 400
        
        # Basic word count validation
        word_count = len(text_content.split())
        if word_count > 1000:
            return jsonify(error="invalid_input", 
                         message=f"Text too long: {word_count} words (max 1000)"), 400
        
        # Smart processing with HTML detection
        try:
            process_result = smart_process_text_input(text_content, primary, spath)
            
            if process_result['success']:
                saved.append(process_result['filename'])
                
                # Save enhanced metadata
                extra_metadata = {
                    'processed_as': process_result['processed_as'],
                    'is_html': process_result.get('is_html', False),
                    'is_sql': process_result.get('is_sql', False),
                    'file_type': process_result.get('file_type', 'text'),       
                    'detected_formats': process_result.get('detected_formats', []),
                    'warnings': process_result.get('warnings', [])
                }
                
                if process_result.get('is_html'):
                    save_session_metadata(spath, "html_paste", extra_metadata=extra_metadata)
                elif process_result.get('is_python'):
                    save_session_metadata(spath, "python_paste", extra_metadata=extra_metadata)
                 
                elif process_result.get('is_sql'):
                    save_session_metadata(spath, "sql_paste", extra_metadata=extra_metadata)
        
    
                else:
                    save_session_metadata(spath, "text", text_content, extra_metadata)
                
                logger.info(f"Created session {sid}: processed as {process_result['processed_as']}")
                
            else:
                logger.error(f"Failed to process text: {process_result.get('error')}")
                return jsonify(error="processing_error", 
                             message="Failed to process text input"), 500

        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            return jsonify(error="server_error", 
                         message="Failed to process text input"), 500
    
    else:
        # Handle file uploads (existing logic remains the same)
        files = request.files.getlist("html_files")
        logger.info(f"[UPLOAD] Received {len(files)} files from request")
        for i, f in enumerate(files):
            logger.info(f"[UPLOAD] File {i}: filename='{f.filename}', size={f.content_length}")
    
    
        if not files or not any(f.filename for f in files):
            return jsonify(error="invalid_input", message="No files uploaded"), 400

        for f in files:
            if f.filename:
                filename_lower = f.filename.lower()
                allowed_extensions = (".html", ".sql", ".py", ".pyw", ".jinja", ".jinja2", ".j2")
                if filename_lower.endswith(allowed_extensions):
              
                    name = secure_filename(f.filename)
                    f.save(str(upload_dir / name))
                    saved.append(name)
                    logger.info(f"[UPLOAD] Saved file: {name}")
                else:
                    logger.warning(f"[UPLOAD] Skipped unsupported file type: {f.filename}")
          
        if not saved:
            return jsonify(error="invalid_input", message="No valid files"), 400
        
        save_session_metadata(spath, "file")
        logger.info(f"Created file session {sid}: {len(saved)} HTML files")

    if hasattr(current_app, "progress_tracker"):
        current_app.progress_tracker.update(sid, 1, "Starting content extraction…")

    executor.submit(
        run_pipeline_async,
        current_app._get_current_object(),
        sid,
        saved,
        primary,
        secondary,
        target,
        enable_refinement,
        refinement_mode,
    )

    return jsonify(
        success=True,
        session_id=sid,
        redirect=f"/processing/{sid}",
        file_count=len(saved),
        input_type=input_type
    )

# ──────────────────── API: progress (polling) ────────────────────────────────
@bp.route("/api/progress/<session_id>")
def api_progress(session_id: str):
    try:
        status = current_app.progress_tracker.get_status(session_id)
        return jsonify(status)
    except Exception as exc:
        logger.exception("Progress polling error for %s", sanitize_for_log(session_id))
        return jsonify(error=str(exc)), 500

# ──────────────────── API: progress (SSE) ────────────────────────────────────
@bp.route("/api/progress-stream/<session_id>")
def api_progress_stream(session_id: str):
    tracker = current_app.progress_tracker

    def generate():
        try:
            for data in tracker.event_stream(session_id):  # already SSE-formatted
                yield data
        except GeneratorExit:
            logger.info("Client closed SSE for %s", sanitize_for_log(session_id))
        except Exception as exc:
            logger.exception("SSE error for %s", sanitize_for_log(session_id))
            yield sse_format("error", json.dumps({"error": str(exc)}))

    return Response(generate(), mimetype="text/event-stream")


@bp.route("/api/regenerate/<session_id>", methods=["POST"])
def api_regenerate(session_id: str):
    """Regenerate final HTML file with user's edited translations"""
    if not session.get("authenticated"):
        return jsonify(error="Authentication required"), 401
    
    if not hasattr(current_app, "session_manager"):
        return jsonify(error="Session manager not available"), 500
    
    try:
        # Get session path
        spath = current_app.session_manager.get_session_path(session_id)
        
        # Get form data
        translation_type = request.form.get("translation_type")  # 'deepl' or 'openai'
        updated_blocks_json = request.form.get("updated_blocks")
        
        if not translation_type or not updated_blocks_json:
            return jsonify(error="Missing required parameters"), 400
        
        if translation_type not in ['deepl', 'openai']:
            return jsonify(error="Invalid translation type"), 400
        
        # Parse updated blocks
        try:
            updated_blocks = json.loads(updated_blocks_json)
        except json.JSONDecodeError:
            return jsonify(error="Invalid JSON format"), 400
        
        logger.info(f"[Regenerate] Processing {sanitize_for_log(translation_type)} for session {sanitize_for_log(session_id)}")
        logger.info(f"[Regenerate] Updated blocks: {list(updated_blocks.keys())}")
        
        # Find the HTML file (should be only one for text sessions)
        uploads_dir = spath / "uploads"
        html_files = list(uploads_dir.glob("*.html")) if uploads_dir.exists() else []
        
        if not html_files:
            return jsonify(error="No HTML files found"), 404
        
        # Use the first HTML file
        html_file = html_files[0]
        basename = html_file.stem
        
        
        # Determine which translation file to update
        if translation_type == 'deepl':
            segments_file = spath / "translated" / basename / "segments_only.json"
        else:  # openai
            segments_file = spath / "refined" / basename / "openai_translations.json"
        
        if not segments_file.exists():
            return jsonify(error=f"Original {translation_type} translation file not found"), 404
        
        # Load original translation file
        with open(segments_file, 'r', encoding='utf-8') as f:
            original_segments = json.load(f)
        
        logger.info(f"[Regenerate] Original segments: {len(original_segments)}")
        
        # Update segments with user's edits
        # Replace all segments for blocks that were edited
        updated_segments = original_segments.copy()
        
        # Get all block numbers that were edited
        edited_block_numbers = set()
        for block_key in updated_blocks.keys():
            # FIX: Use re.match instead of string.match
            match = re.match(r'BLOCK_(\d+)', block_key)
            if match:
                edited_block_numbers.add(int(match.group(1)))
        
        # Remove all segments for edited blocks
        for block_num in edited_block_numbers:
            keys_to_remove = [key for key in updated_segments.keys() 
                            if key.startswith(f"BLOCK_{block_num}_")]
            for key in keys_to_remove:
                del updated_segments[key]
        
        # Add the new edited segments
        for block_key, edited_text in updated_blocks.items():
    # Clean the text content - remove any block identifiers that might be in the text
            clean_text = edited_text
            MAX_TEXT_LENGTH = 100000
            if len(clean_text) > MAX_TEXT_LENGTH:
                logger.warning(f"[Regenerate] Text for {sanitize_for_log(block_key)} truncated from {len(clean_text)} to {MAX_TEXT_LENGTH} characters")
                clean_text = clean_text[:MAX_TEXT_LENGTH]
            
            # Remove block identifiers - using a loop to avoid backtracking
            while True:
                 # Find next BLOCK pattern
                 match = re.search(r'BLOCK_\d+(?:_S\d+)?', clean_text)
                 if not match:
                     break
                 # Find surrounding whitespace manually (no regex backtracking)
                 start = match.start()
                 end = match.end()
                 # Expand start backwards to include whitespace
                 while start > 0 and clean_text[start-1] in ' \t\n\r':
                     start -= 1
                 # Expand end forwards to include whitespace
                 while end < len(clean_text) and clean_text[end] in ' \t\n\r':
                     end += 1
                 # Replace the entire region with a single space
                 clean_text = clean_text[:start] + ' ' + clean_text[end:]
            clean_text = ' '.join(clean_text.split())    
            if clean_text:
                updated_segments[block_key] = clean_text
                logger.info(f"[Regenerate] Added clean segment {sanitize_for_log(block_key)}: {sanitize_for_log(clean_text,50)}")
        
        logger.info(f"[Regenerate] Updated segments: {len(updated_segments)}")
        
        # Save updated segments to a temporary file
        temp_segments_file = segments_file.parent / f"segments_edited_{int(time.time())}.json"
        with open(temp_segments_file, 'w', encoding='utf-8') as f:
            json.dump(updated_segments, f, indent=2, ensure_ascii=False)

        # Get session metadata to determine target language
        metadata = get_session_metadata(spath)
        
        # Try to determine target language from session or use default
        target_lang = "fr"  # Default fallback
        
        # Prepare paths for merging
        non_translatable_html = spath / "extracted" / basename / "non_translatable.html"
        
        # 🔥 FIX: Don't specify exact filename - let step4_merge.py generate it
        output_dir = spath / "results"
        output_dir.mkdir(exist_ok=True)
        
        if not non_translatable_html.exists():
            return jsonify(error="Non-translatable HTML template not found"), 404
        
        # Use existing step4_merge.py logic - MATCH PipelineRunner approach
        script_path = Path(__file__).parent.parent / "core_scripts" / "step4_merge.py"
        
        logger.info(f"[Regenerate] Looking for script at: {sanitize_for_log(script_path)}")
        
        if not script_path.exists():
    
            return jsonify(error="Merge script not found"), 500
        
        # 🔥 FIX: Use dynamic output filename and search for actual generated file
        if translation_type == 'deepl':
            # DeepL-only merge arguments
            temp_output_file = output_dir / f"temp_final_deepl_edited.html"
            cmd = ["python3", str(script_path)]
            cmd.extend([
                "--html", str(non_translatable_html),
                "--deepl", str(temp_segments_file),
                "--output-deepl", str(temp_output_file),
                "--target-lang", target_lang.lower()
            ])
        else:
            # OpenAI merge - we need both original deepl and edited openai files
            original_deepl_file = spath / "translated" / basename / "segments_only.json"
            if not original_deepl_file.exists():
                return jsonify(error="Original DeepL translation file required for OpenAI merge"), 404
            
            temp_output_file = output_dir / f"temp_final_openai_edited.html"
            temp_deepl_output = output_dir / f"temp_final_deepl_unused.html"  # Required by script
            cmd = ["python3", str(script_path)]
            cmd.extend([
                "--html", str(non_translatable_html),
                "--deepl", str(original_deepl_file),      # Original DeepL segments
                "--openai", str(temp_segments_file),      # Edited OpenAI segments  
                "--output-deepl", str(temp_deepl_output), # Required by script
                "--output-openai", str(temp_output_file), # We want this output
                "--target-lang", target_lang.lower()
            ])
        
        logger.info(f"[Regenerate] Merge command: {' '.join(cmd)}")
        
        # Execute merge command
        import subprocess
        try:
            result = subprocess.run(
                cmd,
                cwd=spath,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"[Regenerate] Merge completed successfully")
                logger.info(f"[Regenerate] Stdout: {result.stdout}")
                
                # Clean up temp file
                if temp_segments_file.exists():
                    temp_segments_file.unlink()
                
                # 🔥 FIX: Search for the ACTUAL generated file pattern
                # step4_merge.py generates files like: final_deepl_edited-fr.html
                actual_output_patterns = [
                    f"final_{translation_type}_edited-{target_lang.lower()}.html",
                    f"final_{translation_type}_edited.html",
                    f"temp_final_{translation_type}_edited-{target_lang.lower()}.html",
                    f"temp_final_{translation_type}_edited.html",
                    f"final_{translation_type}-{target_lang.lower()}.html",  # Alternative pattern
                    f"final_{translation_type}.html"  # Alternative pattern
                ]
                
                actual_output_file = None
                for pattern in actual_output_patterns:
                    potential_file = output_dir / pattern
                    if potential_file.exists():
                        actual_output_file = potential_file
                        
                        break
                
                # If specific patterns don't work, search for any recently created HTML files
                if not actual_output_file:
                    
                    for html_file in output_dir.glob("*.html"):
                        # Check if file was modified in the last 60 seconds
                        if time.time() - html_file.stat().st_mtime < 60:
                            actual_output_file = html_file
                            
                            break
                
                if actual_output_file and actual_output_file.exists():
                    
                    
                    # Return the file for download with a clean name
                    return send_from_directory(
                        actual_output_file.parent,
                        actual_output_file.name,
                        as_attachment=True,
                        download_name=f"final_{translation_type}_edited.html"
                    )
                else:
                    logger.error(f"[Regenerate] No output file found in {sanitize_for_log(output_dir)}")
                    # List all files for debugging
                    all_files = list(output_dir.glob("*"))
                    
                    return jsonify(error="Output file was not created"), 500
                    
            else:
                logger.error(f"[Regenerate] Merge failed (exit code {result.returncode})")
                logger.error(f"[Regenerate] Stderr: {result.stderr}")
                logger.error(f"[Regenerate] Stdout: {result.stdout}")
                
                # Clean up temp file
                if temp_segments_file.exists():
                    temp_segments_file.unlink()
                return jsonify(error=f"Merge process failed: {result.stderr}"), 500
                
        except subprocess.TimeoutExpired:
            logger.error(f"[Regenerate] Merge process timed out")
            # Clean up temp file
            if temp_segments_file.exists():
                temp_segments_file.unlink()
            return jsonify(error="Merge process timed out"), 500
        except Exception as e:
            logger.error(f"[Regenerate] Subprocess error: {e}")
            # Clean up temp file
            if temp_segments_file.exists():
                temp_segments_file.unlink()
            return jsonify(error=f"Process execution error: {str(e)}"), 500
        
    except Exception as e:
        logger.error(f"[Regenerate] Error: {e}")
        logger.exception("Full traceback:")
        return jsonify(error=f"Server error: {str(e)}"), 500

# ───────────────────────── File download API ────────────────────────────────
def _send_zip(session_id: str, filename: str):
    if hasattr(current_app, "session_manager"):
        spath = current_app.session_manager.get_session_path(session_id)
        rdir  = spath / "results"
        fpath = rdir / filename
        
        logger.info(f"[DEBUG] Looking for file at: {fpath.absolute()}")
        logger.info(f"[DEBUG] File exists: {fpath.exists()}")
        logger.info(f"[DEBUG] Directory contents: {list(rdir.glob('*')) if rdir.exists() else 'DIR NOT FOUND'}")
        logger.info(f"[DEBUG] Session path: {sanitize_for_log(spath)}")
        logger.info(f"[DEBUG] Results dir exists: {rdir.exists()}")
        
        if filename.endswith(".zip") and fpath.exists():
            return send_from_directory(rdir, filename, as_attachment=True)
    return jsonify(error="File not found"), 404

@bp.route("/api/download/<session_id>/<filename>")
def api_download(session_id: str, filename: str):
    """Preserve old API path."""
    return _send_zip(session_id, filename)

@bp.route("/download/<session_id>/<filename>")
def ui_download(session_id: str, filename: str):
    """Path used inside results.html links."""
    return _send_zip(session_id, filename)

# ───────────────────────── Metrics API ────────────────────────────────
@bp.route("/api/metrics/<session_id>")
def api_metrics(session_id: str):
    """Get consolidated metrics for a session"""
    try:
        if hasattr(current_app, "session_manager"):
            spath = current_app.session_manager.get_session_path(session_id)
            metrics_file = spath / "results" / "consolidated_metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                return jsonify(metrics_data)
            else:
                return jsonify(error="Metrics not found"), 404
        else:
            return jsonify(error="Session manager not available"), 500
    except Exception as exc:
        logger.exception("Metrics API error for %s", sanitize_for_log(session_id))
        return jsonify(error=str(exc)), 500

# ───────────────────────── Debug Routes ────────────────────────────
@bp.route("/debug/check-tmp")
def debug_check_tmp():
    """Quick check of /tmp/ directory"""
    if not session.get("authenticated"):
        return redirect(url_for("main.login"))
    
    results = {
        "tmp_exists": os.path.exists("/tmp/"),
        "tmp_contents": [],
        "zip_files": [],
        "recent_files": []
    }
    
    try:
        tmp_path = Path("/tmp/")
        
        # List all contents
        if tmp_path.exists():
            results["tmp_contents"] = [item.name for item in tmp_path.iterdir()]
            
            # Find ZIP files
            zip_files = list(tmp_path.glob("*.zip"))
            results["zip_files"] = [str(zf) for zf in zip_files]
            
            # Find recent files (last 2 hours)
            for item in tmp_path.iterdir():
                try:
                    if item.is_file():
                        mtime = item.stat().st_mtime
                        if time.time() - mtime < 7200:  # 2 hours
                            results["recent_files"].append({
                                "name": item.name,
                                "path": str(item),
                                "size": item.stat().st_size,
                                "modified_minutes_ago": int((time.time() - mtime) / 60)
                            })
                except:
                    pass
                    
            # Also check for any session-related directories
            session_dirs = [item for item in tmp_path.iterdir() 
                          if item.is_dir() and "session" in item.name.lower()]
            results["session_directories"] = [str(sd) for sd in session_dirs]
            
    except Exception as e:
        results["error"] = str(e)
    
    return jsonify(results) 
