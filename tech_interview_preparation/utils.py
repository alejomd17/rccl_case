"""
Utility functions for EPAM Data Scientist Interview Preparation Sprint

Usage:
    from utils import *
    
    # Start a timed round
    timer = InterviewTimer(minutes=15)
    timer.start()
    
    # Load questions
    questions = load_questions('general_questions.md')
    
    # Score answer
    score = score_answer(user_answer, expected_answer)
"""

import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import re


# ============================================================================
# TIMER & SCHEDULING
# ============================================================================

class InterviewTimer:
    """Timed interview rounds (mimics real interview conditions)"""
    
    def __init__(self, minutes: int = 15, verbose: bool = True):
        self.total_seconds = minutes * 60
        self.start_time = None
        self.verbose = verbose
        self.paused_time = 0
        self.is_running = False
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.is_running = True
        if self.verbose:
            print(f"⏱️  Timer started: {self.total_seconds // 60} minutes")
    
    def elapsed(self) -> int:
        """Get elapsed seconds"""
        if not self.is_running or self.start_time is None:
            return 0
        return int(time.time() - self.start_time - self.paused_time)
    
    def remaining(self) -> int:
        """Get remaining seconds"""
        return max(0, self.total_seconds - self.elapsed())
    
    def is_time_up(self) -> bool:
        """Check if time is up"""
        return self.remaining() <= 0
    
    def status(self) -> str:
        """Get formatted status"""
        elapsed = self.elapsed()
        remaining = self.remaining()
        minutes_e = elapsed // 60
        seconds_e = elapsed % 60
        minutes_r = remaining // 60
        seconds_r = remaining % 60
        
        if self.is_time_up():
            return f"⏰ TIME UP! (elapsed: {minutes_e}:{seconds_e:02d})"
        else:
            return f"⏱️  Elapsed: {minutes_e}:{seconds_e:02d} | Remaining: {minutes_r}:{seconds_r:02d}"
    
    def stop(self):
        """Stop the timer"""
        self.is_running = False
        if self.verbose:
            print(f"Timer stopped. Total time: {self.elapsed()}s")


# ============================================================================
# QUESTION MANAGEMENT
# ============================================================================

def load_questions(filepath: str) -> List[Dict[str, str]]:
    """Load questions from markdown file"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    questions = []
    
    # Parse Q1, Q2, etc.
    pattern = r'### Q(\d+):(.*?)\n\*\*(.*?)\*\*\n\n(.*?)(?=### Q|\Z)'
    matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        q_num, title, metadata, body = match.groups()
        questions.append({
            'number': int(q_num),
            'title': title.strip(),
            'content': body.strip()[:200] + '...',  # Preview
            'full_content': body.strip()
        })
    
    return questions


def get_random_questions(questions: List[Dict], n: int = 5) -> List[Dict]:
    """Get n random questions"""
    import random
    return random.sample(questions, min(n, len(questions)))


def get_questions_by_round(round_num: int) -> List[Dict]:
    """Get questions for a specific round"""
    round_ranges = {
        1: (1, 5),      # Q1-Q5
        2: (6, 15),     # Q6-Q15
        3: (16, 30),    # Q16-Q30
        4: (31, 40),    # Q31-Q40
    }
    
    if round_num not in round_ranges:
        print(f"❌ Invalid round: {round_num}. Valid: 1, 2, 3, 4")
        return []
    
    start, end = round_ranges[round_num]
    filepath = 'general_questions.md'
    all_questions = load_questions(filepath)
    
    return [q for q in all_questions if start <= q['number'] <= end]


# ============================================================================
# ANSWER SCORING
# ============================================================================

def score_answer(user_answer: str, expected_topics: List[str], 
                max_score: int = 10) -> Tuple[int, Dict]:
    """
    Score a user answer based on expected topics covered
    
    Args:
        user_answer: User's response
        expected_topics: List of key topics that should be mentioned
        max_score: Maximum score (default 10)
    
    Returns:
        (score, details_dict)
    """
    score = 0
    covered = []
    missed = []
    
    user_lower = user_answer.lower()
    
    for topic in expected_topics:
        if topic.lower() in user_lower:
            covered.append(topic)
            score += max_score / len(expected_topics)
        else:
            missed.append(topic)
    
    # Bonus for length (coherent answers are longer)
    if len(user_answer.split()) > 50:
        score = min(max_score, score + 1)
    
    return int(score), {
        'score': int(score),
        'max_score': max_score,
        'coverage': len(covered) / len(expected_topics) * 100 if expected_topics else 0,
        'covered_topics': covered,
        'missed_topics': missed
    }


def provide_feedback(score_details: Dict) -> str:
    """Generate feedback based on score details"""
    score = score_details['score']
    coverage = score_details['coverage']
    
    if score >= 9:
        feedback = "🟢 Excellent answer! Strong understanding."
    elif score >= 7:
        feedback = "🟡 Good answer. Consider mentioning: " + ", ".join(score_details['missed_topics'][:2])
    elif score >= 5:
        feedback = "🟠 Acceptable. Missing key concepts: " + ", ".join(score_details['missed_topics'])
    else:
        feedback = "🔴 Needs improvement. Review core concepts."
    
    return feedback


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class InterviewProgress:
    """Track progress through interview rounds"""
    
    def __init__(self):
        self.rounds_completed = []
        self.current_round = None
        self.total_score = 0
        self.questions_answered = 0
    
    def start_round(self, round_num: int, total_questions: int):
        """Start a new round"""
        self.current_round = {
            'round': round_num,
            'total_questions': total_questions,
            'answers': [],
            'start_time': datetime.now(),
            'scores': []
        }
    
    def log_answer(self, question_num: int, user_answer: str, score: int):
        """Log an answer"""
        if self.current_round is None:
            print("❌ No round in progress")
            return
        
        self.current_round['answers'].append({
            'question': question_num,
            'answer': user_answer,
            'score': score
        })
        self.current_round['scores'].append(score)
        self.total_score += score
        self.questions_answered += 1
    
    def end_round(self):
        """End current round and save"""
        if self.current_round is None:
            print("❌ No round in progress")
            return
        
        self.current_round['end_time'] = datetime.now()
        self.current_round['duration'] = (
            self.current_round['end_time'] - self.current_round['start_time']
        ).total_seconds()
        self.current_round['round_score'] = sum(self.current_round['scores'])
        self.current_round['avg_score'] = (
            self.current_round['round_score'] / len(self.current_round['scores'])
            if self.current_round['scores'] else 0
        )
        
        self.rounds_completed.append(self.current_round)
        self.current_round = None
    
    def get_summary(self) -> Dict:
        """Get overall progress summary"""
        return {
            'total_questions_answered': self.questions_answered,
            'total_score': self.total_score,
            'average_score': self.total_score / self.questions_answered if self.questions_answered > 0 else 0,
            'rounds_completed': len(self.rounds_completed),
            'completion_rate': len(self.rounds_completed) / 4 * 100  # 4 rounds total
        }
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("INTERVIEW PROGRESS SUMMARY")
        print("="*60)
        print(f"Questions answered: {summary['total_questions_answered']}")
        print(f"Total score: {summary['total_score']}")
        print(f"Average score: {summary['average_score']:.1f}/10")
        print(f"Rounds completed: {summary['rounds_completed']}/4")
        print(f"Completion: {summary['completion_rate']:.0f}%")
        print("="*60)


# ============================================================================
# SCHEDULE MANAGEMENT
# ============================================================================

SPRINT_SCHEDULE = {
    1: {
        'day': 'Thursday',
        'time': '21:00-22:00',
        'questions': 'Q1-Q5',
        'topics': ['ML Basics', 'Overfitting', 'Bias-Variance'],
    },
    2: {
        'day': 'Friday',
        'time': '14:00-15:00',
        'questions': 'Q6-Q15',
        'topics': ['Model Selection', 'Data Processing', 'Clustering'],
    },
    3: {
        'day': 'Monday',
        'time': '17:00-18:00',
        'questions': 'Q16-Q30',
        'topics': ['Deep Learning', 'CV', 'NLP', 'Time Series'],
    },
    4: {
        'day': 'Tuesday',
        'time': '12:45-15:00',
        'questions': 'Q31-Q40',
        'topics': ['Advanced', 'Production', 'Scenarios'],
    },
}


def get_schedule() -> Dict:
    """Get sprint schedule"""
    return SPRINT_SCHEDULE


def print_schedule():
    """Print sprint schedule"""
    print("\n" + "="*60)
    print("EPAM DATA SCIENTIST INTERVIEW PREP SPRINT")
    print("="*60)
    for round_num, details in SPRINT_SCHEDULE.items():
        print(f"\n🎯 ROUND {round_num}: {details['day']} {details['time']}")
        print(f"   Questions: {details['questions']}")
        print(f"   Topics: {', '.join(details['topics'])}")
    print("\n" + "="*60)


def get_next_round() -> Optional[Tuple[int, Dict]]:
    """Get next upcoming round (based on current time)"""
    # This is a placeholder - in real use, compare with actual datetime
    return 1, SPRINT_SCHEDULE[1]


# ============================================================================
# HELPERS
# ============================================================================

def format_answer(answer: str, max_length: int = 500) -> str:
    """Format answer for display"""
    if len(answer) > max_length:
        return answer[:max_length] + "..."
    return answer


def print_question(question: Dict):
    """Pretty print a question"""
    print(f"\n{'='*60}")
    print(f"Q{question['number']}: {question['title']}")
    print(f"{'='*60}")
    print(question['full_content'][:500] + "...")
    print(f"{'='*60}\n")


def validate_answer(answer: str, min_words: int = 20) -> Tuple[bool, str]:
    """Validate answer meets minimum criteria"""
    word_count = len(answer.split())
    
    if word_count < min_words:
        return False, f"Answer too short ({word_count} words, need ≥{min_words})"
    
    if len(answer) < 50:
        return False, "Answer too brief"
    
    return True, "✅ Answer valid"


def save_progress(progress: InterviewProgress, filepath: str = 'progress.json'):
    """Save progress to JSON file"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'progress': progress.get_summary(),
        'rounds': progress.rounds_completed,
        'total_score': progress.total_score,
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"✅ Progress saved to {filepath}")


def load_progress(filepath: str = 'progress.json') -> Optional[Dict]:
    """Load progress from JSON file"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================================
# MAIN - Example Usage
# ============================================================================

if __name__ == '__main__':
    print("🎓 EPAM Interview Prep Utilities")
    print("Import this module to use the functions:")
    print("\n  from utils import *")
    print("\nAvailable:")
    print("  - InterviewTimer: timed rounds")
    print("  - InterviewProgress: track answers")
    print("  - load_questions: load from markdown")
    print("  - score_answer: evaluate responses")
    print("  - get_schedule: view sprint schedule")
    print("\nExample:")
    print("  timer = InterviewTimer(15)")
    print("  timer.start()")
    print("  print(timer.status())")
