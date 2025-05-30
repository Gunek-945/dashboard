# User profile data for the API
DEFAULT_USER_PROFILE = {
    "name": "Alex Doe",
    "alias": "MayaUser123",
    "countryOfOrigin": "Germany",
    "dateOfRegistration": "2024-07-01",
    "email": "alex.doe@example.com",
    "phone": "+49 123 4567890",
    "bio": "A proactive individual seeking new opportunities and settling into a new environment. Keen on technology and community engagement.",
    "onboardingSummary": """
    During onboarding, Alex expressed a primary need for employment assistance, specifically in the tech sector. 
    They also highlighted the importance of understanding local healthcare options and managing official paperwork efficiently. 
    Alex has a background in software development and is eager to integrate into the local community.
    Key needs identified: Job search, Healthcare navigation, Document management, Local orientation.
    """,
    "age": 34,
    "gender": "Female",
    "dateOfBirth": "1990-05-15",
    "challenges": ["Finding suitable housing", "Learning the local language", "Navigating bureaucracy"]
}

# Action items data (excluding workflow-related items as requested)
SUGGESTED_ACTIONS_DATA = [
    {
        "id": "action1",
        "title": "Register for Local Health Insurance",
        "status": "Not Started",
        "details": {
            "what": "Complete and submit the application form for public health insurance.",
            "why": "Essential for accessing healthcare services in your new country.",
            "preparedByMaya": "Basic personal information pre-filled in the application form (Form H-123). Digital copy of passport prepared.",
            "stillMissing": "Proof of address (e.g., rental agreement), Bank details for premium payments.",
            "links": [
                {"text": "Official Health Insurance Portal", "url": "#"},
                {"text": "FAQ about Health Insurance", "url": "#"}
            ],
            "documents": ["Form H-123 (pre-filled).pdf", "Passport Scan.pdf"],
            "deadlines": "Within 3 months of registration.",
        },
        "icon": "ShieldCheckIcon",
    },
    {
        "id": "action2",
        "title": "Apply for a Tax Identification Number",
        "status": "In Progress",
        "progressValue": 60,
        "details": {
            "what": "Submit the request for a personal tax ID to the local tax office.",
            "why": "Required for employment and managing financial affairs.",
            "preparedByMaya": "Application form (Form T-45) partially filled. List of required documents compiled.",
            "stillMissing": "Certified copy of birth certificate (if applicable), Signature on Form T-45.",
            "links": [{"text": "Tax Office Information", "url": "#"}],
            "deadlines": "As soon as possible, ideally before starting employment.",
        },
        "icon": "IdentificationIcon",
    },
    {
        "id": "action3",
        "title": "Open a Local Bank Account",
        "status": "Completed",
        "details": {
            "what": "Visit a local bank branch or use an online service to open a current account.",
            "why": "Necessary for receiving salary, paying bills, and daily transactions.",
            "preparedByMaya": "Comparison of local banks provided. Checklist of typical documents needed.",
            "stillMissing": "Completed during previous interactions. Account details are now securely stored.",
            "documents": ["Bank Comparison Sheet.pdf", "Bank Account Opening Checklist.pdf"],
        },
        "icon": "CreditCardIcon",
    },
    {
        "id": "action4",
        "title": "Prepare for Job Interview",
        "status": "In Progress",
        "progressValue": 40,
        "details": {
            "what": "Research common interview questions, practice answers, and prepare questions to ask the interviewer.",
            "why": "Increases confidence and chances of a successful interview outcome.",
            "preparedByMaya": "List of common behavioral questions compiled. Company research links provided.",
            "stillMissing": "Tailored answers to specific job descriptions. List of personal strengths and weaknesses.",
            "links": [{"text": "Tips for Job Interviews", "url": "#"}],
            "deadlines": "Ongoing, intensify before scheduled interviews.",
        },
        "icon": "BriefcaseIcon",
    },
    {
        "id": "action5",
        "title": "Register with Local Municipality",
        "status": "Not Started",
        "details": {
            "what": "Complete residence registration at the local municipal office.",
            "why": "Legal requirement for establishing residency and accessing local services.",
            "preparedByMaya": "Registration form template prepared. List of required documents compiled.",
            "stillMissing": "Proof of address, completed registration form, appointment booking.",
            "links": [{"text": "Municipal Services Portal", "url": "#"}],
            "deadlines": "Within 14 days of moving to new address.",
        },
        "icon": "BuildingOfficeIcon",
    },
    {
        "id": "action6",
        "title": "Apply for Public Transportation Pass",
        "status": "Not Started",
        "details": {
            "what": "Apply for a monthly or annual public transportation pass.",
            "why": "Cost-effective way to travel around the city for work and daily activities.",
            "preparedByMaya": "Comparison of different pass types and pricing. Application form links provided.",
            "stillMissing": "Passport photo, proof of address, payment method setup.",
            "links": [{"text": "Transportation Authority", "url": "#"}],
            "deadlines": "No strict deadline, but recommended within first month.",
        },
        "icon": "TruckIcon",
    },
    {
        "id": "action7",
        "title": "Set Up Internet and Utilities",
        "status": "In Progress",
        "progressValue": 25,
        "details": {
            "what": "Arrange internet, electricity, and water services for your residence.",
            "why": "Essential services for daily living and remote work capabilities.",
            "preparedByMaya": "Provider comparison sheet created. Contact information for major providers compiled.",
            "stillMissing": "Service provider selection, contract signing, installation scheduling.",
            "links": [{"text": "Utility Providers Comparison", "url": "#"}],
            "deadlines": "As soon as possible after moving in.",
        },
        "icon": "WifiIcon",
    },
    {
        "id": "action8",
        "title": "Find Local Healthcare Provider",
        "status": "Not Started",
        "details": {
            "what": "Research and register with a local general practitioner (GP).",
            "why": "Important for ongoing healthcare needs and emergency situations.",
            "preparedByMaya": "List of nearby healthcare providers. Information about the local healthcare system.",
            "stillMissing": "Provider selection, appointment booking, medical history transfer.",
            "links": [{"text": "Healthcare Provider Directory", "url": "#"}],
            "deadlines": "Within first 2 months of arrival.",
        },
        "icon": "HeartIcon",
    }
]

DOCUMENTS_DATA = [
    {"id": "doc1", "name": "Form H-123 (pre-filled).pdf", "type": "PDF", "dateAdded": "2024-07-10", "size": "1.2 MB", "url": "#"},
    {"id": "doc2", "name": "Passport Scan.pdf", "type": "PDF", "dateAdded": "2024-07-05", "size": "800 KB", "url": "#"},
    {"id": "doc3", "name": "Bank Comparison Sheet.pdf", "type": "PDF", "dateAdded": "2024-07-08", "size": "450 KB", "url": "#"},
    {"id": "doc4", "name": "Referral Letter - Job Agency.docx", "type": "DOCX", "dateAdded": "2024-07-15", "size": "150 KB", "url": "#"},
    {"id": "doc5", "name": "Onboarding Notes.txt", "type": "TXT", "dateAdded": "2024-07-01", "size": "10 KB", "url": "#"},
    {"id": "doc6", "name": "Municipal Registration Form.pdf", "type": "PDF", "dateAdded": "2024-07-12", "size": "650 KB", "url": "#"},
    {"id": "doc7", "name": "Transportation Pass Application.pdf", "type": "PDF", "dateAdded": "2024-07-14", "size": "320 KB", "url": "#"},
    {"id": "doc8", "name": "Utility Providers Comparison.pdf", "type": "PDF", "dateAdded": "2024-07-16", "size": "890 KB", "url": "#"},
] 