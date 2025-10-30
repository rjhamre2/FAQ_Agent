#!/usr/bin/env python3
import requests
import json

# Test data
train_data = {
    "user_id": "test_user001",
    "content": """Introducing NimbleAI: Your AI Chat Assistant
NimbleAI is an AI-powered chat assistant designed to help businesses reply instantly to customers across WhatsApp, Instagram, and websites. It leverages advanced AI language models to understand natural language, respond based on your FAQs and policies, and is a major step up from older, rule-based chatbots.

Core Features and Capabilities

Instant Replies on All Channels: NimbleAI seamlessly integrates with WhatsApp Business, Instagram DMs, and your website chat via an easy-to-install widget.

Intelligent Answering: It can be trained on your FAQs and product information (like materials, sizing, and availability) to answer complex product and policy questions.

Contextual & Multilingual: Unlike older bots, NimbleAI understands context, supports English and Hindi out of the box, and can handle a wide variety of queries without manual scripting.

Lead Generation & Escalation: The bot automatically collects leads (emails/phone numbers) and, if it can't answer a query or a human is needed, it will escalate the conversation to your support team or create a follow-up ticket.

Customization: You can fully customize the responses and the tone of voice to perfectly match your brand.

Ideal Users & Business Impact

Any business with online customer inquiries can benefit, especially D2C brands, travel/logistics companies, and e-commerce stores.

Time Savings: It typically handles 70–80% of repetitive questions, saving your team hours daily.

Faster Responses: Brands on average see response times improve by 60–80% (some by over 70%!).

Setup and Support

Quick Setup: Most businesses can get started within a day. No developer is needed; the setup is simple, and the NimbleAI team provides full onboarding assistance.

Data Security: Your data is securely encrypted and stored on cloud servers, compliant with major data privacy regulations.

Pricing: Pricing is flexible, based on your business size and usage volume (either message volume or monthly active users). A free trial is available.

Ready to see it in action? Book a demo or start your free trial.""",
    "additionalProp1": {}
}

# Test questions
test_questions = [
    "What is NimbleAI?",
    "What are the core features of NimbleAI?",
    "How much does NimbleAI cost?",
    "What languages does NimbleAI support?",
    "How can I get started with NimbleAI?",
    "What is the free trial?",
    "How does NimbleAI handle lead generation?",
    "What types of businesses can benefit from NimbleAI?"
]

def test_train():
    print("🚀 Testing /train endpoint...")
    try:
        response = requests.post("http://localhost:8000/train", json=train_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_ask(question):
    print(f"\n❓ Testing /ask with: '{question}'")
    try:
        ask_data = {
            "question": question,
            "user_id": "test_user001",
            "comp_name": "NimbleAI",
            "specialization": "AI chatbots",
            "sender_name": "Test User",
            "sender_number": "+1234567890",
            "time_stamp": "2024-01-01 10:00:00"
        }
        response = requests.post("http://localhost:8000/ask", json=ask_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result.get('answer', 'No answer found')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("🧪 Testing FAQ Agent Endpoints")
    print("=" * 50)
    
    # Test training
    if test_train():
        print("\n✅ Training successful!")
        
        # Test asking questions
        print("\n🔍 Testing /ask endpoint with various questions...")
        for question in test_questions:
            test_ask(question)
            print("-" * 30)
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()

