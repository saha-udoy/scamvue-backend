from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import re
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ScamVue API")

# Make sure this route is at the top, right after creating the app
@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "ScamVue API is running", "status": "ok"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

# Update CORS for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
classifier = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli")

class Message(BaseModel):
    message: str

def check_phishing_indicators(text):
    # Common phishing patterns
    phishing_patterns = {
        'urgency': r'urgent|immediate|quickly|asap|emergency|limited time|act now|expires?|deadline|today\?|right now',
        'credentials': r'password|login|credential|account|verify|confirmation|security|authenticate',
        'financial': r'bank|credit.?card|\$|money|payment|transfer|wallet|bitcoin|crypto|eth|investment',
        'personal_info': r'ssn|social security|address|birth.?date|passport|license|id.?number',
        'suspicious_links': r'click|link|url|website|site|http|bit\.ly|tiny\.cc|sign.?in',
        'threats': r'suspend|disabled|closed|blocked|terminated|locked|restricted|limited',
        'prizes': r'won|winner|prize|reward|gift|congratulations|lucky|selected',
        'pressure': r'only|limited|exclusive|special.?offer|one.?time|today.?only',
        # New patterns
        'unsolicited_contact': r'recruiter|gave.+contact|referred|mentioned you|that\'s why I contacted|reaching out|got your contact',
        'job_bait': r'job opportunity|work opportunity|position|career|hiring|recruitment|employment',
        'time_pressure': r'today|would you be available|discuss|share with you|opportunity',
        'impersonation': r'I am \w+|my name is|this is \w+|representing|on behalf',
        'vague_opportunity': r'information to share|discuss the work|opportunity|details|interested in|potential'
    }
    
    matches = {}
    for category, pattern in phishing_patterns.items():
        if re.search(pattern, text.lower()):
            matches[category] = True
    
    # Check for suspicious combinations
    if 'unsolicited_contact' in matches and 'job_bait' in matches:
        matches['suspicious_recruitment'] = True
    if 'time_pressure' in matches and 'vague_opportunity' in matches:
        matches['pressure_tactics'] = True
    
    return matches

def check_spam_indicators(text):
    # Common spam patterns
    spam_patterns = {
        'marketing': r'buy|sell|discount|offer|deal|price|sale|shop|store|product',
        'promotion': r'free|bonus|extra|save|discount|\d+% off',
        'mass_marketing': r'subscribe|unsubscribe|newsletter|mailing.?list',
        'excessive_punctuation': r'!{2,}|\?{2,}',
        'all_caps': r'[A-Z]{4,}',
        'repetitive': r'(.)\1{2,}',
        'spam_words': r'weight.?loss|earn|income|work.?from.?home|make.?money|business.?opportunity'
    }
    
    matches = {}
    for category, pattern in spam_patterns.items():
        if re.search(pattern, text.lower()):
            matches[category] = True
    
    return matches

@app.post("/analyze")
async def analyze_message(message: Message):
    try:
        text = message.message
        
        # Check for phishing and spam indicators
        phishing_indicators = check_phishing_indicators(text)
        spam_indicators = check_spam_indicators(text)
        
        # Initial classification with more specific labels
        candidate_labels = [
            "normal friendly conversation",
            "casual chat message",
            "business communication",
            "phishing attempt",
            "spam advertisement",
            "scam message",
            "unsolicited job offer",
            "suspicious recruitment message"
        ]
        
        result = classifier(
            text,
            candidate_labels,
            hypothesis_template="This message is a {}."
        )
        
        # Enhanced risk assessment
        risk_level = "legitimate"
        confidence = result['scores'][0]
        primary_label = result['labels'][0]
        
        # Count indicators
        num_phishing = len(phishing_indicators)
        num_spam = len(spam_indicators)
        
        # Specific checks for recruitment scams
        recruitment_indicators = {
            'unsolicited_contact', 'job_bait', 'time_pressure', 
            'vague_opportunity', 'suspicious_recruitment', 'pressure_tactics'
        }
        recruitment_matches = recruitment_indicators.intersection(phishing_indicators.keys())
        
        # Determine risk based on both ML classification and pattern matching
        if len(recruitment_matches) >= 2:
            risk_level = "phishing"
            confidence = max(confidence, 0.85)
        elif num_phishing >= 2 or "phishing" in primary_label or "scam" in primary_label:
            risk_level = "phishing"
            confidence = max(confidence, 0.8 if num_phishing >= 3 else 0.6)
        elif num_spam >= 2 or "spam" in primary_label or "advertisement" in primary_label:
            risk_level = "spam"
            confidence = max(confidence, 0.7 if num_spam >= 3 else 0.5)
        
        # Additional checks for high-risk combinations
        if 'unsolicited_contact' in phishing_indicators and 'vague_opportunity' in phishing_indicators:
            risk_level = "phishing"
            confidence = max(confidence, 0.9)
        
        # Format indicators for display
        detected_indicators = []
        if phishing_indicators:
            detected_indicators.extend(phishing_indicators.keys())
        if spam_indicators:
            detected_indicators.extend(spam_indicators.keys())
        
        return {
            "risk": risk_level,
            "confidence": confidence,
            "primary_indicator": primary_label,
            "secondary_indicator": ", ".join(detected_indicators[:3]),
            "details": {
                "phishing_indicators": list(phishing_indicators.keys()),
                "spam_indicators": list(spam_indicators.keys()),
                "ml_classification": result['labels'][0],
                "ml_confidence": result['scores'][0]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup")
    logger.info("Initializing ML model...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 