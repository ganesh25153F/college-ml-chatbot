from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os

print("🚀 Training NECG ML Chatbot...")

# Create bot
bot = ChatBot('NECG_Bot')

# Your college data
conversations = [
    "hi", "Namaste macha! Welcome to NECG Chatbot! 🤖",
    "hello", "Hello! chepu mama",
    
    # Faculty
    "cse faculty", "🎓 CSE: Dr. Venkateswara Rao (HOD), Mrs. Saritha (Python), Mrs. Surekha (Java)",
    "ece faculty", "📡 ECE: Prof. Neeraja Reddy (HOD), Dr. BMK (Circuits)",
    "python", "📚 Python: Mrs. Saritha (8 yrs exp)",
    "java", "☕ Java: Mrs. Surekha (8 yrs exp)",
    
    # Portals
    "results", "📊 Results: https://narayanagroup.co.in/patient/student/EngLogin.aspx",
    "attendance", "👤 Attendance: http://115.241.194.20/patient/student/student_login.aspx",
    "fees", "💳 Fees: https://narayanagroup.co.in/onlinefeepay/razorpay/createordernew.aspx",
    "admissions", "🎓 Admissions: https://ap-btech-admissions-3.onrender.com/",
    
    # Fees/Courses
    "cse fees", "💰 CSE B.Tech: ₹1,40,000/year",
    "what courses", "📚 B.Tech CSE/ECE, MCA. 95% placements!"
]

trainer = ListTrainer(bot)
trainer.train(conversations)

print("✅ ML Training COMPLETE!")
print("💾 Model saved. Run 'python app.py' next!")
