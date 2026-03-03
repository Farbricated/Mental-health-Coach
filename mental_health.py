"""
Mindful Pro v5.0 — Multi-Profile Mental Health Companion
Profiles: General · Military/Veterans · First Responders · Mental Health Professionals
          Healthcare Workers · Students · Caregivers

Validated tools: PHQ-9 · GAD-7 · PCL-5 · PSS-4 · ProQOL · MBI · Columbia Safety Screen
Safety Planning · Psychoeducation Library · Moral Injury · Compassion Fatigue

Run:
    pip install streamlit plotly requests python-dotenv
    streamlit run mental_health.py
"""

import streamlit as st
import requests
import os
import sqlite3
import random
from datetime import datetime, timedelta, date
from typing import Dict, List
from collections import Counter, defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="Mindful Pro",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PROFILES — The core of the multi-audience design
# ═══════════════════════════════════════════════════════════════════════════════

PROFILES = {
    "general": {
        "label": "General Public",
        "icon": "🌿",
        "color": "#6fa8dc",
        "desc": "Everyday stress, anxiety, mood, and wellbeing",
        "language": {
            "greeting_suffix": "",
            "distress_label": "overwhelmed",
            "talk_intro": "Share what's on your mind. There's no right way to start.",
            "coping_header": "Grounding & calming tools",
        },
        "crisis_lines": [
            ("iCall (India)",         "+91-9152987821"),
            ("AASRA (India)",         "+91-9820466726"),
            ("988 Lifeline (USA)",    "988"),
            ("Crisis Text (USA)",     "Text HOME → 741741"),
            ("Samaritans (UK)",       "116 123"),
        ],
        "assessments": ["PHQ-9", "GAD-7", "PSS-4"],
        "psychoed_topics": ["anxiety", "depression", "stress", "sleep", "burnout"],
        "system_prompt_addon": "",
    },
    "military": {
        "label": "Military / Veterans",
        "icon": "🎖",
        "color": "#7a9e7e",
        "desc": "Active duty, veterans, military families",
        "language": {
            "greeting_suffix": " — mission accomplished just by showing up.",
            "distress_label": "under operational stress",
            "talk_intro": "No rank here. Say what's real.",
            "coping_header": "Combat stress & regulation tools",
        },
        "crisis_lines": [
            ("Veterans Crisis Line",   "988 then press 1"),
            ("Veterans Crisis Chat",   "veteranscrisisline.net"),
            ("Military OneSource",     "1-800-342-9647"),
            ("Give an Hour",           "giveanhour.org"),
            ("SAMHSA (USA)",           "1-800-662-4357"),
        ],
        "assessments": ["PHQ-9", "PCL-5", "PSS-4"],
        "psychoed_topics": ["ptsd", "moral_injury", "hypervigilance", "transition", "mst"],
        "system_prompt_addon": """
This user is military/veteran. Key adaptations:
- Use direct, clear language — avoid clinical jargon
- Frame struggle through operational lens: 'stress response', not 'disorder'
- Acknowledge moral injury as distinct from PTSD when relevant
- Never use stigma-laden language; military culture values strength — reframe help-seeking AS strength
- Reference peer support (battle buddy concept) where appropriate
- Be aware of transition stress (civilian reintegration) and identity challenges
- MST (Military Sexual Trauma) is a sensitive topic — never probe, but hold space""",
    },
    "first_responder": {
        "label": "First Responders",
        "icon": "🚨",
        "color": "#c97b84",
        "desc": "Police, firefighters, paramedics, dispatchers",
        "language": {
            "greeting_suffix": " — it's okay to not be okay.",
            "distress_label": "dealing with operational stress",
            "talk_intro": "What you carry doesn't stay on scene. You can say it here.",
            "coping_header": "Critical incident stress tools",
        },
        "crisis_lines": [
            ("Safe Call Now (USA)",     "1-206-459-3020"),
            ("Code Green Campaign",     "codegreencampaign.org"),
            ("First H.E.L.P.",          "firsthelp.org.au"),
            ("Badge of Life",           "badgeoflife.com"),
            ("988 Lifeline",            "988"),
        ],
        "assessments": ["PHQ-9", "PCL-5", "PSS-4"],
        "psychoed_topics": ["ptsd", "cisd", "hypervigilance", "vicarious_trauma", "moral_injury"],
        "system_prompt_addon": """
This user is a first responder (police/fire/EMS/dispatcher). Key adaptations:
- Peer-to-peer tone — like a fellow responder who's been there
- Validate the culture: strength, duty, self-reliance are real values, not obstacles
- Critical incident stress debriefing (CISD) concepts may be relevant
- Cumulative stress (daily low-level exposure) is as important as acute incidents
- Dispatcher stress is often invisible — validate it explicitly if relevant
- Avoid anything that sounds like a formal HR/clinical evaluation
- Operational fatigue and moral injury are common; name them when appropriate""",
    },
    "mental_health_pro": {
        "label": "Mental Health Professional",
        "icon": "🧠",
        "color": "#9b8ec4",
        "desc": "Therapists, counselors, psychologists, psychiatrists",
        "language": {
            "greeting_suffix": "",
            "distress_label": "experiencing vicarious trauma or burnout",
            "talk_intro": "You hold space for others every day. This is space for you.",
            "coping_header": "Secondary trauma & self-care tools",
        },
        "crisis_lines": [
            ("Psych-Armor",             "psycharmor.org"),
            ("APA Psychologist Locator","locator.apa.org"),
            ("Physician Support Line",  "1-888-409-0141"),
            ("988 Lifeline",            "988"),
            ("Crisis Text",             "Text HOME → 741741"),
        ],
        "assessments": ["PHQ-9", "GAD-7", "ProQOL"],
        "psychoed_topics": ["vicarious_trauma", "burnout", "compassion_fatigue", "secondary_stress", "self_care_pro"],
        "system_prompt_addon": """
This user is a mental health professional. Key adaptations:
- Peer-collegial tone — they have clinical literacy, don't over-explain basic concepts
- Their primary risks: vicarious trauma, compassion fatigue, secondary traumatic stress, burnout
- Acknowledge the paradox: the most skilled at supporting others often resist seeking help themselves
- Supervisory needs, countertransference, and boundary fatigue may arise — hold space for these
- May carry grief from client losses or crises — this is real occupational grief
- Encourage professional supervision as a resource, not a referral away
- Respect clinical autonomy — don't tell them what they already know; be curious with them""",
    },
    "healthcare": {
        "label": "Healthcare Worker",
        "icon": "🏥",
        "color": "#5ba8a0",
        "desc": "Doctors, nurses, allied health, support staff",
        "language": {
            "greeting_suffix": "",
            "distress_label": "experiencing burnout or moral distress",
            "talk_intro": "You care for everyone. This is where someone cares for you.",
            "coping_header": "Moral distress & burnout tools",
        },
        "crisis_lines": [
            ("Physician Support Line",  "1-888-409-0141"),
            ("Nurse Support (USA)",     "1-800-662-4357"),
            ("Doctors in Distress (UK)","docindistress.org.uk"),
            ("iCall (India)",           "+91-9152987821"),
            ("988 Lifeline",            "988"),
        ],
        "assessments": ["PHQ-9", "GAD-7", "ProQOL", "MBI"],
        "psychoed_topics": ["burnout", "moral_distress", "compassion_fatigue", "vicarious_trauma", "moral_injury"],
        "system_prompt_addon": """
This user is a healthcare worker. Key adaptations:
- Acknowledge systemic stressors — understaffing, moral distress, impossible demands are real
- Moral distress (knowing the right thing but being unable to do it) is central to healthcare burnout
- Do not minimize with 'self-care'; many systemic issues are not solved by yoga
- Compassion fatigue and depersonalization are occupational hazards — normalize them
- Grief load: healthcare workers carry cumulative patient loss rarely processed
- Pandemic-era trauma may still be active
- Encourage peer support and professional resources specific to healthcare workers""",
    },
    "student": {
        "label": "Student",
        "icon": "📚",
        "color": "#d4a843",
        "desc": "University, graduate school, professional programs",
        "language": {
            "greeting_suffix": "",
            "distress_label": "overwhelmed by academic or life pressure",
            "talk_intro": "No judgment here. Say what's actually going on.",
            "coping_header": "Academic stress & focus tools",
        },
        "crisis_lines": [
            ("Crisis Text (USA)",       "Text HOME → 741741"),
            ("988 Lifeline (USA)",      "988"),
            ("iCall (India)",           "+91-9152987821"),
            ("The Trevor Project",      "1-866-488-7386"),
            ("Samaritans (UK)",         "116 123"),
        ],
        "assessments": ["PHQ-9", "GAD-7", "PSS-4"],
        "psychoed_topics": ["anxiety", "depression", "burnout", "imposter_syndrome", "sleep"],
        "system_prompt_addon": """
This user is a student. Key adaptations:
- Relatable tone — acknowledge the real pressures: grades, money, identity, future uncertainty
- Imposter syndrome is extremely common in academic environments — name it when relevant
- Academic perfectionism and people-pleasing often drive anxiety and depression here
- Student finances, housing instability, and social isolation are real mental health risk factors
- Acknowledge that asking for help in academic culture can feel like admitting failure — it's not
- Graduate students specifically: isolation, advisor dynamics, and uncertain career paths are unique stressors
- Be warm and non-clinical; many students are accessing mental health support for the first time""",
    },
    "caregiver": {
        "label": "Caregiver",
        "icon": "🤲",
        "color": "#c4906e",
        "desc": "Family caregivers, parents, those supporting loved ones",
        "language": {
            "greeting_suffix": "",
            "distress_label": "carrying the weight of caregiving",
            "talk_intro": "The care you give to others matters. So does the care you give to yourself.",
            "coping_header": "Caregiver burnout & respite tools",
        },
        "crisis_lines": [
            ("Caregiver Action Network", "caregiveraction.org"),
            ("NAMI Helpline",            "1-800-950-6264"),
            ("988 Lifeline",             "988"),
            ("iCall (India)",            "+91-9152987821"),
            ("Crisis Text",              "Text HOME → 741741"),
        ],
        "assessments": ["PHQ-9", "GAD-7", "ProQOL"],
        "psychoed_topics": ["compassion_fatigue", "caregiver_burnout", "grief", "boundaries", "respite"],
        "system_prompt_addon": """
This user is a caregiver (family or informal). Key adaptations:
- Validate the invisible and undervalued nature of caregiving labor
- Caregiver guilt is extremely common — when they feel relief, resentment, or wish it would end
- Anticipatory grief (caring for someone with a terminal or declining illness) needs specific acknowledgment
- Isolation is a primary caregiver mental health risk — social withdrawal is often not chosen
- Encourage respite without dismissing barriers (cost, availability, guilt)
- Many caregivers have completely submerged their own identity — hold space for this
- Compassion fatigue in caregivers looks different from professionals — it's often silent and denied""",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATED CLINICAL ASSESSMENTS
# ═══════════════════════════════════════════════════════════════════════════════

PHQ9_ITEMS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you're a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed — or the opposite, being so fidgety or restless",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
]
PHQ9_OPTIONS = ["Not at all", "Several days", "More than half the days", "Nearly every day"]

GAD7_ITEMS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it's hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]

PSS4_ITEMS = [
    "How often have you been upset because of something that happened unexpectedly?",
    "How often have you felt unable to control the important things in your life?",
    "How often have you felt confident about your ability to handle personal problems? (reverse scored)",
    "How often have you felt that things were going your way? (reverse scored)",
]
PSS4_OPTIONS = ["Never", "Almost never", "Sometimes", "Fairly often", "Very often"]

PCL5_ITEMS = [
    "Repeated, disturbing, and unwanted memories of the stressful experience",
    "Repeated, disturbing dreams of the stressful experience",
    "Suddenly feeling or acting as if the stressful experience were actually happening again",
    "Feeling very upset when something reminded you of the stressful experience",
    "Having strong physical reactions when something reminded you of the stressful experience",
    "Avoiding memories, thoughts, or feelings related to the stressful experience",
    "Avoiding external reminders of the stressful experience",
    "Trouble remembering important parts of the stressful experience",
    "Having strong negative beliefs about yourself, other people, or the world",
    "Blaming yourself or someone else for the stressful experience or what happened after it",
    "Having strong negative feelings such as fear, horror, anger, guilt, or shame",
    "Loss of interest in activities that you used to enjoy",
    "Feeling distant or cut off from other people",
    "Trouble experiencing positive feelings",
    "Irritable behavior, angry outbursts, or acting aggressively",
    "Taking too many risks or doing things that could cause you harm",
    "Being 'superalert' or watchful or on guard",
    "Feeling jumpy or easily startled",
    "Having difficulty concentrating",
    "Trouble falling or staying asleep",
]
PCL5_OPTIONS = ["Not at all", "A little bit", "Moderately", "Quite a bit", "Extremely"]

PROQOL_ITEMS = [  # Simplified 10-item version
    ("cs", "I get satisfaction from being able to help people"),
    ("cs", "I feel energized by my work with those I help"),
    ("cs", "I believe I can make a difference through my work"),
    ("bo", "I feel overwhelmed because my caseload seems endless"),
    ("bo", "I feel trapped by my work as a helper"),
    ("bo", "I feel worn out because of my work as a helper"),
    ("sts", "I am preoccupied with more than one person I help"),
    ("sts", "I have thoughts of traumatic events that my clients have shared"),
    ("sts", "I am losing sleep over traumatic experiences of a person I help"),
    ("sts", "I feel 'on edge' because of my work-related trauma"),
]
PROQOL_OPTIONS = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]

MBI_ITEMS = [  # Simplified burnout screen
    ("ee", "I feel emotionally drained from my work"),
    ("ee", "I feel used up at the end of the workday"),
    ("ee", "I feel fatigued when I get up in the morning and have to face another day on the job"),
    ("ee", "Working with people all day is really a strain for me"),
    ("dp", "I feel I treat some patients/clients as if they were impersonal objects"),
    ("dp", "I've become more callous toward people since I took this job"),
    ("dp", "I worry that this job is hardening me emotionally"),
    ("pa", "I feel I'm positively influencing other people's lives through my work"),
    ("pa", "I feel very energetic in my work"),
    ("pa", "I feel exhilarated after working closely with patients/clients"),
]
MBI_OPTIONS = ["Never", "A few times/year", "Monthly", "A few times/month", "Weekly", "Daily"]

# ═══════════════════════════════════════════════════════════════════════════════
#  PSYCHOEDUCATION LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

PSYCHOED = {
    "ptsd": {
        "title": "Understanding PTSD",
        "subtitle": "Post-Traumatic Stress — what it is and what it isn't",
        "color": "rose",
        "sections": [
            ("What is PTSD?",
             "PTSD is the brain's attempt to protect you. After trauma, the amygdala "
             "(threat-detection center) stays on high alert. This isn't weakness — it's a "
             "normal nervous system response to abnormal events. The problem is that the alarm "
             "keeps firing even when the danger is past."),
            ("PTSD ≠ weakness",
             "Anyone can develop PTSD. Combat veterans, first responders, survivors of accidents, "
             "assault, or childhood trauma. PTSD is not a character flaw. Neuroimaging shows measurable "
             "brain changes — it is a physical injury as much as a broken bone."),
            ("The four clusters",
             "Re-experiencing (flashbacks, nightmares) · Avoidance (staying away from reminders) · "
             "Negative mood & cognition (guilt, numbness, loss of enjoyment) · Hyperarousal "
             "(hypervigilance, startle response, anger, sleep problems). You don't need all four to "
             "be struggling — any combination is valid."),
            ("What helps",
             "Evidence-based treatments include Prolonged Exposure (PE), Cognitive Processing Therapy (CPT), "
             "and EMDR. Medication (SSRIs) can reduce symptoms. Peer support is particularly effective for "
             "military and first responders. You don't have to relive trauma to heal — CPT focuses on "
             "meaning, not re-experiencing."),
            ("When to seek help",
             "If symptoms are interfering with work, relationships, or daily function for more than a month "
             "after a traumatic event, professional support is recommended. The PCL-5 screening tool "
             "in this app can help you track symptoms over time."),
        ],
    },
    "moral_injury": {
        "title": "Moral Injury",
        "subtitle": "Distinct from PTSD — and often misdiagnosed",
        "color": "amber",
        "sections": [
            ("What is moral injury?",
             "Moral injury is damage to your core sense of right and wrong after committing, witnessing, "
             "or failing to prevent acts that violate your deeply held moral beliefs. It is NOT the same as PTSD, "
             "though they often co-occur. PTSD is rooted in fear; moral injury is rooted in shame, guilt, and betrayal."),
            ("Who experiences it?",
             "Military personnel (civilian casualties, orders followed under duress), first responders "
             "(being unable to save someone), healthcare workers (allocating scarce resources, futile treatment), "
             "mental health professionals (client suicide), caregivers (end-of-life decisions)."),
            ("Signs of moral injury",
             "Persistent guilt or shame disproportionate to events · 'I did things I can never forgive' · "
             "Spiritual crisis or loss of meaning · Anger at leadership or 'the system' · "
             "Social withdrawal · Difficulty with intimacy or trust · Feeling permanently changed."),
            ("What helps",
             "Adaptive Disclosure Therapy (developed specifically for military moral injury) · "
             "Meaning-centered approaches · Spiritual/religious support if relevant · "
             "Peer confession (telling your story to someone who was there) · "
             "Acts of repair or reparation where possible. Standard PTSD treatments are less effective for "
             "pure moral injury — naming the distinction matters."),
        ],
    },
    "compassion_fatigue": {
        "title": "Compassion Fatigue",
        "subtitle": "The cost of caring, and how to recover",
        "color": "lavender",
        "sections": [
            ("What it is",
             "Compassion fatigue is the emotional and physical exhaustion that results from the prolonged "
             "stress of helping or wanting to help traumatized people. It's sometimes called 'secondary traumatic "
             "stress' when it's specifically caused by exposure to others' trauma. Unlike burnout, compassion "
             "fatigue can onset suddenly after a single powerful encounter."),
            ("Signs",
             "Reduced empathy for people you used to care deeply about · Dreading going to work · "
             "Intrusive thoughts about clients/patients · Emotional numbness or detachment · "
             "Physical exhaustion despite adequate sleep · Reduced sense of satisfaction from helping · "
             "Cynicism about outcomes ('nothing I do matters')."),
            ("Compassion fatigue vs burnout",
             "Burnout is gradual depletion from chronic workplace stress — workload, autonomy, values mismatch. "
             "Compassion fatigue is specifically triggered by the weight of witnessing and absorbing others' suffering. "
             "You can have both simultaneously. Burnout responds to systemic/organizational change; "
             "compassion fatigue responds more to trauma-processing and meaning restoration."),
            ("Recovery",
             "Deliberate decompression rituals after heavy sessions · Regular supervision or peer consultation · "
             "Clear psychological boundaries (not emotional walls — boundaries) · Physical activity · "
             "Replenishing compassion satisfaction (reconnecting with why you do this work) · "
             "Personal therapy — not optional, essential. The ProQOL assessment helps track your levels."),
        ],
    },
    "burnout": {
        "title": "Burnout",
        "subtitle": "Chronic depletion — not a personal failure",
        "color": "blue",
        "sections": [
            ("The three dimensions",
             "Maslach's model: Emotional exhaustion (depleted emotional resources) · "
             "Depersonalization (emotional distance, cynicism about those you serve) · "
             "Reduced personal accomplishment (feeling ineffective despite effort). "
             "You can score high in one area without the others."),
            ("Burnout is systemic",
             "Burnout is not a sign that you're weak, uncommitted, or wrong for your profession. "
             "Research consistently shows it's driven by systemic factors: excessive workload, "
             "lack of autonomy, insufficient recognition, community breakdown, unfairness, and "
             "values mismatch. Individual solutions (yoga, mindfulness) don't fix structural problems."),
            ("How it progresses",
             "Enthusiasm → Stagnation → Frustration → Apathy. Early warning signs: "
             "Dreading work you used to love · Increased cynicism · Physical symptoms "
             "(headaches, insomnia, immune suppression) · Social withdrawal · Increased errors."),
            ("Recovery",
             "Burnout recovery typically takes 3–6 months of deliberate change. Key elements: "
             "Actual rest (not just weekends) · Addressing root systemic issues where possible · "
             "Reconnecting with personal values and purpose · Social support · "
             "Sometimes: a role change, leave of absence, or new environment. "
             "Therapy focused on identity and values — not just symptoms — is most effective."),
        ],
    },
    "vicarious_trauma": {
        "title": "Vicarious Traumatization",
        "subtitle": "When others' trauma becomes part of you",
        "color": "lavender",
        "sections": [
            ("What it is",
             "Vicarious traumatization (VT) is the cumulative transformation in a helper's inner world "
             "resulting from empathic engagement with trauma material. Unlike compassion fatigue, VT "
             "changes your fundamental worldview — your sense of safety, trust, and meaning shift "
             "permanently. First described in trauma therapists, it also affects journalists, lawyers, "
             "first responders, and researchers."),
            ("How it differs from PTSD",
             "You haven't experienced the trauma directly. But the nervous system doesn't fully "
             "distinguish between witnessed and experienced — mirror neurons create partial "
             "somatic responses. The images, stories, and suffering absorbed over time accumulate "
             "and alter your perception of the world as a safe place."),
            ("Signs specific to VT",
             "Disrupted worldview ('nowhere is safe') · Hypervigilance outside work · "
             "Difficulty with intimacy and trust even in safe relationships · "
             "Intrusive imagery from client/patient material · Loss of hope or spiritual crisis · "
             "Inability to tolerate ambiguity · Over-identification with those you help."),
            ("Transformation, not just recovery",
             "The goal isn't to return to who you were — exposure to others' trauma can also create "
             "post-traumatic growth: expanded capacity for empathy, deeper meaning, changed priorities. "
             "The key is processing rather than accumulation. Supervision, peer consultation, "
             "trauma-informed self-care, and regular reflection are protective."),
        ],
    },
    "hypervigilance": {
        "title": "Hypervigilance",
        "subtitle": "When your threat system won't stand down",
        "color": "rose",
        "sections": [
            ("What it is",
             "Hypervigilance is a state of heightened alertness in which you're constantly scanning "
             "for threats. For military personnel and first responders, it's a trained, adaptive skill. "
             "The problem comes when it doesn't turn off — when you're back home but your nervous "
             "system is still 'on patrol.'"),
            ("The physiology",
             "Chronic activation of the sympathetic nervous system (fight/flight) without adequate "
             "parasympathetic recovery (rest/digest) leads to: elevated cortisol, disrupted sleep, "
             "cardiovascular strain, irritability, impaired digestion, and emotional dysregulation. "
             "Your body is paying a physiological tax for constant readiness."),
            ("In civilian life",
             "Sitting with your back to the wall in restaurants · Scanning exits · "
             "Startling at normal sounds · Interpreting neutral faces as threatening · "
             "Difficulty in crowded places · Trouble relaxing even in objectively safe situations · "
             "Relationship conflict from irritability or emotional unavailability."),
            ("Down-regulation tools",
             "Physiological sigh (double inhale through nose, long exhale) — fastest known method "
             "to reduce acute stress · Cold water on wrists · Extended exhales · "
             "Progressive muscle relaxation (consciously tensing then releasing each muscle group) · "
             "Somatic therapy and EMDR are highly effective for nervous-system level healing. "
             "These aren't weakness — they're maintenance, like reloading."),
        ],
    },
    "moral_distress": {
        "title": "Moral Distress in Healthcare",
        "subtitle": "Knowing the right thing but being unable to do it",
        "color": "teal",
        "sections": [
            ("Definition",
             "Moral distress occurs when you know the ethically right action but are constrained "
             "from taking it — by institutional barriers, resource limitations, hierarchical dynamics, "
             "or policy. Andrew Jameton first described it in nursing in 1984. It's now recognized "
             "across all healthcare professions and is a primary driver of healthcare burnout."),
            ("Common scenarios",
             "Continuing aggressive treatment you believe is causing suffering · Inadequate staffing "
             "forcing care rationing · Following orders you believe are wrong · Allocating scarce "
             "resources (ICU beds, organs, medications) · Being unable to spend adequate time with "
             "patients due to systemic demands. The injury is the moral violation, not the busyness."),
            ("The residue",
             "'Moral residue' — the lingering sense of moral taint after repeated compromises — "
             "accumulates over a career. Each incident alone may seem manageable. Cumulatively, "
             "they alter your relationship to your profession, your sense of integrity, and your "
             "capacity for moral agency. It is a form of moral injury."),
            ("Institutional vs individual response",
             "Moral distress requires institutional response — ethics consultation, shared decision-making, "
             "staffing investment. Individual tools help: moral residue processing (writing/speaking about "
             "specific incidents), ethics team engagement, peer support. If you're carrying this alone, "
             "you're solving a systems problem with personal resources. Both matter."),
        ],
    },
    "cisd": {
        "title": "Critical Incident Stress",
        "subtitle": "Normal reactions to abnormal events",
        "color": "rose",
        "sections": [
            ("What is a critical incident?",
             "Any situation that causes unusually strong emotional reactions that interfere with "
             "normal functioning — mass casualty events, child victims, line-of-duty deaths, "
             "situations with high personal investment. The normal reaction includes intrusive "
             "thoughts, sleep disruption, and emotional intensity. This is not PTSD; it's a normal "
             "acute stress response."),
            ("CISM — Critical Incident Stress Management",
             "CISM is a structured peer-support program including defusings (brief, same-day support) "
             "and debriefings (CISD — structured group intervention 24-72 hours after an incident). "
             "Note: CISD is most effective as peer support, not as mandatory organizational therapy. "
             "Voluntary peer-led models show the best outcomes."),
            ("What you might experience",
             "Physical: Fatigue, tremors, chest tightness, nausea, disrupted sleep. "
             "Cognitive: Confusion, poor concentration, flashbacks, heightened awareness. "
             "Emotional: Shock, grief, anger, guilt, feeling numb, fear. "
             "Behavioral: Withdrawal, irritability, appetite changes, increased alcohol use."),
            ("Recovery",
             "Most first responders recover naturally within 4-6 weeks with peer support. "
             "Key: Don't isolate · Talk to someone who understands (peer support) · "
             "Maintain normal routine where possible · Allow yourself to feel without judgment · "
             "Monitor alcohol — it masks symptoms and delays recovery. "
             "If symptoms persist beyond 4-6 weeks or worsen, PCL-5 screening and professional support are warranted."),
        ],
    },
    "imposter_syndrome": {
        "title": "Imposter Syndrome",
        "subtitle": "Why high achievers feel like frauds",
        "color": "amber",
        "sections": [
            ("What it is",
             "Imposter syndrome is the persistent internal experience of believing you are not as competent "
             "as others perceive you to be — and that you'll eventually be 'found out.' First described "
             "by Clance & Imes (1978) in high-achieving women; now recognized across genders, cultures, "
             "and professions. It's remarkably common in academic environments."),
            ("The patterns",
             "The Perfectionist (sets impossibly high standards) · The Superwoman/man (works harder "
             "than everyone to prove worth) · The Natural Genius (if it's not effortless, I must be stupid) · "
             "The Soloist (asking for help feels like admitting failure) · "
             "The Expert (needs to know everything before feeling qualified)."),
            ("Why academia amplifies it",
             "Competitive selection processes convince you that 'real' students belong here and you "
             "don't · Evaluation culture means you're constantly being judged · High-achieving peers "
             "trigger social comparison · The shift from 'best in your class' to 'one among many excellent people' "
             "is genuinely disorienting · Graduate programs in particular offer little positive feedback."),
            ("What helps",
             "Name it — just knowing it has a name reduces its power. "
             "Reality test: list your actual accomplishments. What would you tell a friend who said these things? "
             "Normalize failure and learning curves — they are the curriculum, not evidence of inadequacy. "
             "Talk to peers — you'll find you're not alone. Therapy (especially ACT) helps with the "
             "underlying perfectionism and self-worth beliefs."),
        ],
    },
    "caregiver_burnout": {
        "title": "Caregiver Burnout",
        "subtitle": "When the well runs dry — and how to refill it",
        "color": "orange",
        "sections": [
            ("The invisible labor",
             "Family caregiving is largely invisible, undervalued, and isolating. Whether you're caring "
             "for a parent with dementia, a partner with chronic illness, or a child with complex needs, "
             "the emotional, physical, and financial toll is immense. Caregiver burnout is not weakness — "
             "it's the predictable result of giving beyond sustainable limits."),
            ("The guilt trap",
             "Caregiver guilt is nearly universal: feeling resentment, wishing it would end, feeling "
             "relief at small moments of freedom — and then feeling terrible about feeling those things. "
             "These feelings are normal. They don't mean you love the person less. "
             "They mean you're human under enormous pressure."),
            ("Warning signs",
             "Withdrawing from friends and other loved ones · Losing interest in activities you once enjoyed · "
             "Feeling blue, irritable, hopeless, and helpless · Changes in appetite, weight, or sleep · "
             "Getting sick more often · Feeling that caregiving is your entire identity · "
             "Increasing feelings of resentment toward the person you care for."),
            ("Sustainable caregiving",
             "Respite is not abandonment — it is maintenance. You cannot pour from an empty vessel. "
             "Key practices: Identify even one hour per week that belongs to you · "
             "Accept help rather than managing alone (specific asks work better than 'let me know if you need anything') · "
             "Join a caregiver support group — peer understanding is irreplaceable · "
             "Consider professional support; caregiver depression is treatable."),
        ],
    },
    "transition": {
        "title": "Military-to-Civilian Transition",
        "subtitle": "Identity, purpose, and belonging after service",
        "color": "green",
        "sections": [
            ("More than a job change",
             "Military transition is one of the most psychologically complex life changes a person can make. "
             "It's not just changing careers — it's changing identity, community, culture, purpose, and "
             "structure simultaneously. The loss of the military community, clear hierarchy, and mission "
             "focus can create profound disorientation even for those who wanted to leave."),
            ("Common experiences",
             "Loss of identity ('I don't know who I am without the uniform') · "
             "Loss of community (the intensity of military bonds is rarely replicated in civilian life) · "
             "Mission void (civilian work can feel purposeless or trivial by comparison) · "
             "Hypervigilance in civilian contexts · Difficulty with civilians' perceived entitlement or complaints · "
             "Financial and career uncertainty · Relationship strain."),
            ("The civilian culture gap",
             "Military culture prizes directness, hierarchy, unit cohesion, and mission. "
             "Civilian culture often feels indirect, politically navigated, and individualistic. "
             "Neither is wrong — but the translation takes time and causes real friction. "
             "Difficulty in civilian workplaces does not mean you failed."),
            ("Building a new mission",
             "Research on veteran wellbeing consistently shows: purpose, community, and service "
             "are the core protective factors. Finding work that connects to a larger mission · "
             "Building a peer community (veteran service orgs, community orgs) · "
             "Therapy from a veteran-aware clinician · Patience with yourself — transition takes 2-4 years on average."),
        ],
    },
    "mst": {
        "title": "Military Sexual Trauma",
        "subtitle": "Resources and support",
        "color": "lavender",
        "sections": [
            ("Recognition",
             "Military Sexual Trauma (MST) refers to sexual assault or repeated, threatening sexual "
             "harassment that occurred in the military. MST is widespread across genders — affecting "
             "both men and women — and is associated with high rates of PTSD, depression, and substance use. "
             "It is never the victim's fault."),
            ("Unique barriers",
             "Reporting within a chain of command creates unique barriers: fear of retaliation, "
             "not being believed, career consequences, or being forced to continue serving alongside the perpetrator. "
             "These systemic realities make MST different from assault in civilian contexts. "
             "Your response to an impossible situation was reasonable."),
            ("VA support",
             "Veterans can access MST-related care through the VA regardless of whether they reported the MST, "
             "whether it occurred once or repeatedly, or how long ago it happened. "
             "Every VA medical center has an MST Coordinator. No documentation is required."),
            ("Support",
             "VA MST Coordinator: va.gov/find-locations · "
             "Veterans Crisis Line: 988 then press 1 · "
             "RAINN: 1-800-656-4673 · "
             "Safe Helpline (DoD): 1-877-995-5247 · "
             "Male Survivor (for men): malesurvivor.org"),
        ],
    },
    "anxiety": {
        "title": "Understanding Anxiety",
        "subtitle": "What it is, why it persists, and what actually helps",
        "color": "blue",
        "sections": [
            ("The function of anxiety",
             "Anxiety is your threat-detection system working. The amygdala fires before the prefrontal cortex "
             "(thinking brain) can evaluate whether the threat is real. This is evolutionarily useful — "
             "hesitating to assess a tiger is fatal. The problem is that modern threats (emails, deadlines, "
             "social judgment) trigger the same biological response as physical danger."),
            ("The anxiety cycle",
             "Trigger → Anxious thought → Physical activation (heart rate, muscle tension, breathing changes) → "
             "Avoidance or safety behavior → Short-term relief → Reinforcement of avoidance → "
             "Increased sensitivity to trigger. Avoidance is the fuel that keeps anxiety alive long-term."),
            ("What works",
             "CBT (specifically, exposure-based approaches) has the strongest evidence base. The key insight: "
             "you cannot think your way out of anxiety — you have to act your way through it. "
             "Controlled exposure to feared situations (without avoidance) teaches the brain "
             "that the threat is manageable. Medication (SSRIs, SNRIs) combined with therapy is more "
             "effective than either alone for moderate-severe anxiety."),
            ("What doesn't work long-term",
             "Reassurance-seeking · Avoiding feared situations · 'White knuckling' through without processing · "
             "Alcohol or substance use to reduce physical symptoms (short-term relief, long-term amplification). "
             "The nervous system learns from experience — only new experiences can update old learning."),
        ],
    },
    "depression": {
        "title": "Understanding Depression",
        "subtitle": "More than sadness — and what actually helps",
        "color": "blue",
        "sections": [
            ("What depression actually is",
             "Clinical depression is not sadness. It's a disorder of motivation, energy, and reward "
             "processing — the inability to feel pleasure from things that used to bring it (anhedonia). "
             "Many people with depression don't cry; they feel nothing. The brain's reward circuitry "
             "is disrupted at a neurological level."),
            ("Why 'just do something' is hard",
             "Depression is uniquely self-defeating: the activities that would help (exercise, socializing, "
             "meaningful work) are precisely the things depression makes impossible. Behavioral activation "
             "theory addresses this — small, structured activities begin to restore reward circuitry, "
             "but the activation must come before the motivation, not after."),
            ("What works",
             "Behavioral activation (do first, feel later) is highly effective even without full motivation. "
             "CBT (challenging the cognitive distortions depression creates). Medication (antidepressants) "
             "for moderate-severe depression — typically 4-6 weeks to take effect. "
             "Exercise has comparable efficacy to antidepressants for mild-moderate depression. "
             "Social connection, even when unwanted, protects against severity."),
            ("Red flags requiring immediate attention",
             "Thoughts of suicide or self-harm · Complete inability to function · Psychotic symptoms · "
             "Inability to eat or sleep for extended periods. If these are present, same-day "
             "professional contact is important. PHQ-9 item 9 specifically screens for suicidal ideation."),
        ],
    },
    "sleep": {
        "title": "Sleep & Mental Health",
        "subtitle": "The bidirectional relationship most people underestimate",
        "color": "lavender",
        "sections": [
            ("The relationship",
             "Sleep disruption is both a symptom and a cause of most mental health conditions. "
             "Poor sleep increases anxiety reactivity by up to 60% (studies measuring amygdala response). "
             "After just one night of poor sleep, the prefrontal cortex (rational thinking, emotion regulation) "
             "is significantly less effective. This is not a moral failing — it's neuroscience."),
            ("Common culprits",
             "Screen light (blue wavelength suppresses melatonin) · Variable sleep/wake times "
             "(destroys circadian rhythm) · Caffeine half-life is 5-7 hours — afternoon coffee "
             "is still active at midnight · Alcohol (disrupts REM sleep, causes rebound awakening) · "
             "Lying in bed awake (trains the brain that bed = wakefulness)."),
            ("CBT-I — the gold standard",
             "Cognitive Behavioral Therapy for Insomnia (CBT-I) is more effective than sleep medication "
             "and doesn't carry dependency risks. Core techniques: stimulus control (bed = sleep only), "
             "sleep restriction therapy (counterintuitive but powerful), relaxation training, "
             "and cognitive restructuring of catastrophic sleep thoughts."),
            ("Practical tools tonight",
             "Keep a consistent wake time (even on weekends — this anchors circadian rhythm) · "
             "No screens 60 minutes before bed · If you can't sleep after 20 minutes, leave the bed — "
             "do something calm in dim light until sleepy · 4-7-8 breathing activates parasympathetic "
             "nervous system · Keep your room cool (core body temperature must drop for sleep onset)."),
        ],
    },
    "self_care_pro": {
        "title": "Self-Care for Helpers",
        "subtitle": "What it actually means — and what it doesn't",
        "color": "lavender",
        "sections": [
            ("What self-care actually means",
             "Self-care for helpers is not spa days and yoga (though those aren't bad). It's deliberately "
             "maintaining the systems that allow you to do emotionally demanding work sustainably. "
             "That means: clinical supervision · genuine days off · processing your emotional responses "
             "to clients/patients · boundaries that protect your private life · and access to your own therapy."),
            ("The professional duty dimension",
             "As a mental health professional, your self-care is not optional. Ethics codes across "
             "licensing boards require competence — and burnout, vicarious trauma, and compassion "
             "fatigue impair competence. Self-care is a professional duty, not a luxury."),
            ("Supervision — underused and undervalued",
             "Regular clinical supervision reduces vicarious traumatization significantly. "
             "Many experienced clinicians stop seeking supervision once licensed. This is when "
             "it often becomes most necessary — caseloads are heavier, cases are more complex, "
             "and the isolation of private practice is real. Peer consultation groups are an accessible alternative."),
            ("Personal therapy — the double standard",
             "You'd recommend therapy for a client showing your symptoms. The barrier for clinicians is "
             "often: finding someone outside your professional network, the discomfort of being the patient, "
             "or the belief that you 'should be able to handle this.' None of these barriers are logical. "
             "Most licensing boards and ethics codes explicitly support therapist's own therapy."),
        ],
    },
    "respite": {
        "title": "Respite for Caregivers",
        "subtitle": "Rest is not abandonment — it's maintenance",
        "color": "orange",
        "sections": [
            ("Why respite matters",
             "Caregiver burnout is one of the strongest predictors of care quality declining and "
             "care relationships breaking down. Regular respite — genuine breaks from caregiving — "
             "is not a luxury. Studies show caregivers who take regular breaks provide better care "
             "over a longer period than those who don't. Martyrdom is not sustainable care."),
            ("Barriers and how to address them",
             "Guilt ('how can I take time off when they need me') — Reframe: your wellbeing "
             "is part of their care plan. Cost — Look into: respite care grants, local adult day programs, "
             "ARCH National Respite Network (USA), charity programs. No one to cover — identify "
             "one person in your network and make a specific ask (not 'let me know if you need anything' "
             "but 'can you cover Tuesday morning?')."),
            ("Types of respite",
             "In-home respite (paid or volunteer caregiver comes to you) · Adult day programs (structured "
             "activities in a center setting) · Short-term residential (temporary placement) · "
             "Informal respite (trusted friend, faith community, neighbor) · Virtual — even a phone call "
             "that's not about caregiving, or 30 minutes of personal time, counts."),
            ("What to do with respite time",
             "Whatever refuels you. Not errands. Not caregiving by proxy (worrying). "
             "Something that existed before caregiving became your identity — or something new "
             "that connects you to who you are outside this role. Even small: a walk alone, "
             "a phone call with a friend, sitting in silence without a pager."),
        ],
    },
    "boundaries": {
        "title": "Boundaries in Helping Roles",
        "subtitle": "What boundaries actually are (and aren't)",
        "color": "teal",
        "sections": [
            ("What boundaries are",
             "Boundaries are the limits that define how you can be treated and what you will and won't do. "
             "They are not walls, coldness, or lack of care — they are the conditions under which you "
             "can continue to care. Professional boundaries protect both you and those you serve. "
             "Personal boundaries protect your relationships, health, and identity."),
            ("Why helpers struggle with boundaries",
             "Helping professions often attract people who learned that their value comes from being "
             "needed — not from being. Saying no can feel like abandonment, selfishness, or failure. "
             "Many helpers have experienced their own trauma or dysfunction in which they learned "
             "to suppress their needs. The helper role can unconsciously meet those same needs."),
            ("Signs of boundary violations",
             "Taking work home emotionally every night · Checking messages on days off · "
             "Feeling personally responsible for outcomes beyond your control · "
             "Rescuing rather than supporting · Feeling resentment toward those you help · "
             "Having no time or energy for relationships outside your helping role."),
            ("Building sustainable limits",
             "Identify the line: where does empathy end and over-involvement begin? "
             "Practice the physical act of transitioning — a ritual that marks the end of the work "
             "day (a different route home, changing clothes, a specific phrase). "
             "Say no as a complete sentence — 'I can't take that on right now' without extensive justification. "
             "Remember: a boundary you cannot hold isn't really a boundary."),
        ],
    },
    "grief": {
        "title": "Grief & Loss",
        "subtitle": "Complicated, non-linear, and always valid",
        "color": "blue",
        "sections": [
            ("What grief actually is",
             "Grief is the natural response to any significant loss — not only death. Job loss, relationship "
             "endings, diagnosis, lost futures, role changes, and cumulative small losses all trigger grief. "
             "Kübler-Ross's stages (denial, anger, bargaining, depression, acceptance) were not intended "
             "as a linear sequence — most people move through them non-linearly, cyclically, and in their "
             "own order. There is no 'supposed to' in grief."),
            ("Anticipatory grief",
             "Grief can begin before the loss — while a loved one is still alive but declining, while a "
             "diagnosis is progressing, while a relationship is clearly ending. This is especially common "
             "in caregivers and healthcare workers. It is real grief, deserves real support, and often "
             "occurs in isolation because the loss hasn't 'officially' happened yet."),
            ("Cumulative grief",
             "Repeated losses without adequate processing create accumulation — each new loss lands on "
             "top of unprocessed ones. Healthcare workers, first responders, and therapists who lose clients "
             "are especially vulnerable. Individual losses seem 'manageable.' The weight of many does not."),
            ("What helps",
             "There is no shortcut through grief. The research is clear: you can go around it temporarily "
             "but you cannot skip it. Presence, not problem-solving, is what grievers need from others. "
             "For yourself: naming and expressing (writing, talking, creating) · allowing the waves "
             "without judgment · continuing small, meaningful activities · seeking community · "
             "professional support when grief is complicated (prolonged, debilitating, or traumatic)."),
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL   = "llama-3.1-8b-instant"
    TEMPERATURE  = 0.75
    MAX_TOKENS   = 600
    DB_PATH      = "mindful_pro.db"

    AFFIRMATIONS = [
        "You are stronger than you think.",
        "Progress, not perfection.",
        "Your feelings are valid — and they will pass.",
        "Small steps still move you forward.",
        "You've survived every hard day so far.",
        "Healing is not linear — be patient with yourself.",
        "You deserve the same compassion you give others.",
        "Asking for help is not weakness — it's the most effective strategy available.",
    ]

    CRISIS_UNIVERSAL = [
        ("988 Lifeline (USA/Canada)", "988"),
        ("Crisis Text (USA)",         "Text HOME → 741741"),
        ("iCall (India)",             "+91-9152987821"),
        ("Samaritans (UK)",           "116 123"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css(profile_color="#6fa8dc"):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    :root {{
        --bg:        #0f1117;
        --surface:   #161b27;
        --surface2:  #1d2538;
        --border:    #252f45;
        --text:      #e2e8f4;
        --muted:     #7a8aa0;
        --accent:    {profile_color};
        --green:     #7ec8a0;
        --amber:     #d4a843;
        --rose:      #c97b84;
        --teal:      #5ba8a0;
        --lavender:  #9b8ec4;
        --orange:    #c4906e;
        --r:         14px;
    }}
    html, body, .stApp {{
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], header {{ display: none !important; }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    .main .block-container {{
        padding: 1.8rem 1rem 4rem 1rem !important;
        max-width: 700px !important;
    }}
    h1 {{ font-family:'Lora',Georgia,serif!important;font-weight:400!important;
          color:var(--text)!important;letter-spacing:-0.02em; }}
    h2 {{ font-family:'Lora',serif!important;font-weight:400!important;
          color:var(--text)!important;font-size:1.4rem!important; }}
    h3 {{ font-family:'DM Sans',sans-serif!important;font-weight:500!important;
          color:var(--text)!important;font-size:1.0rem!important; }}
    .card {{
        background:var(--surface);border:1px solid var(--border);
        border-radius:var(--r);padding:16px 20px;margin:8px 0;
    }}
    .card-blue   {{ border-left:3px solid var(--accent); }}
    .card-green  {{ border-left:3px solid var(--green); }}
    .card-amber  {{ border-left:3px solid var(--amber); }}
    .card-rose   {{ border-left:3px solid var(--rose); }}
    .card-teal   {{ border-left:3px solid var(--teal); }}
    .card-lav    {{ border-left:3px solid var(--lavender); }}
    .card-orange {{ border-left:3px solid var(--orange); }}
    .msg-user {{
        background:var(--surface2);border:1px solid var(--border);
        border-radius:18px 18px 4px 18px;padding:12px 16px;
        margin:6px 0 6px 56px;font-size:0.92rem;line-height:1.65;
    }}
    .msg-bot {{
        background:linear-gradient(135deg,#18253f 0%,#1a2d48 100%);
        border:1px solid #2a3d60;border-radius:18px 18px 18px 4px;
        padding:14px 18px;margin:6px 56px 6px 0;
        font-size:0.92rem;line-height:1.75;
    }}
    .msg-label {{ font-size:0.68rem;color:var(--muted);letter-spacing:0.08em;
                  text-transform:uppercase;margin-bottom:5px; }}
    .msg-label-bot {{ color:var(--accent); }}
    .affirmation {{
        text-align:center;font-family:'Lora',serif;font-style:italic;
        font-size:1.0rem;color:#8faacc;padding:10px 16px;margin-bottom:20px;
    }}
    .crisis-strip {{
        background:#1f1218;border:1px solid #6b2a38;
        border-radius:var(--r);padding:14px 18px;margin-bottom:12px;
    }}
    .crisis-line {{
        display:flex;justify-content:space-between;
        padding:5px 0;border-bottom:1px solid #3a1a22;font-size:0.83rem;
    }}
    .crisis-number {{ color:var(--amber);font-weight:500; }}
    .score-bar-wrap {{ background:var(--border);border-radius:4px;height:7px;margin:8px 0; }}
    .score-bar {{ height:7px;border-radius:4px; }}
    .profile-badge {{
        display:inline-flex;align-items:center;gap:6px;
        background:var(--surface2);border:1px solid var(--border);
        border-radius:20px;padding:4px 12px;font-size:0.8rem;color:var(--accent);
    }}
    .m-tile {{
        background:var(--surface);border:1px solid var(--border);
        border-radius:var(--r);padding:14px 16px;text-align:center;
    }}
    .m-val {{ font-family:'Lora',serif;font-size:1.9rem;color:var(--accent);line-height:1; }}
    .m-lbl {{ font-size:0.68rem;color:var(--muted);text-transform:uppercase;
              letter-spacing:0.07em;margin-top:4px; }}
    .prog-wrap {{ background:var(--border);border-radius:4px;height:6px;margin:8px 0; }}
    .prog-fill {{ height:6px;border-radius:4px;
                  background:linear-gradient(90deg,var(--accent),var(--green)); }}
    .stButton > button {{
        background:var(--surface2)!important;color:var(--text)!important;
        border:1px solid var(--border)!important;border-radius:10px!important;
        font-family:'DM Sans',sans-serif!important;font-size:0.87rem!important;
        transition:all 0.15s ease!important;padding:0.38rem 0.9rem!important;
    }}
    .stButton > button:hover {{
        border-color:var(--accent)!important;color:var(--accent)!important;
    }}
    .stTextArea textarea, .stTextInput input {{
        background:var(--surface2)!important;color:var(--text)!important;
        border:1px solid var(--border)!important;border-radius:10px!important;
        font-family:'DM Sans',sans-serif!important;font-size:0.9rem!important;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color:var(--accent)!important;
        box-shadow:0 0 0 2px rgba(111,168,220,0.12)!important;
    }}
    .stSelectbox [data-baseweb="select"] > div {{
        background:var(--surface2)!important;border-color:var(--border)!important;border-radius:10px!important;
    }}
    .stSlider [data-testid="stSlider"] > div > div > div {{
        background:var(--accent)!important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background:var(--surface)!important;border-radius:12px!important;
        border:1px solid var(--border)!important;padding:3px!important;gap:2px!important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background:transparent!important;color:var(--muted)!important;
        border-radius:9px!important;font-size:0.83rem!important;font-family:'DM Sans',sans-serif!important;
    }}
    .stTabs [aria-selected="true"] {{
        background:var(--surface2)!important;color:var(--text)!important;
    }}
    hr {{ border-color:var(--border)!important;margin:18px 0!important; }}
    ::-webkit-scrollbar {{ width:4px; }}
    ::-webkit-scrollbar-thumb {{ background:var(--border);border-radius:2px; }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_db():
    return Database()

class Database:
    def __init__(self):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, role TEXT, content TEXT, session_id TEXT
                );
                CREATE TABLE IF NOT EXISTS mood_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT, mood_score INTEGER, emotion TEXT,
                    note TEXT, energy INTEGER, sleep_hours REAL
                );
                CREATE TABLE IF NOT EXISTS gratitude (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT, entry TEXT
                );
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created TEXT, title TEXT, category TEXT,
                    completed INTEGER DEFAULT 0, completed_date TEXT
                );
                CREATE TABLE IF NOT EXISTS streaks (
                    id INTEGER PRIMARY KEY, last_checkin TEXT,
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    total_days INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT, tool TEXT, score REAL,
                    interpretation TEXT, subscores TEXT, profile TEXT
                );
                CREATE TABLE IF NOT EXISTS safety_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created TEXT, updated TEXT,
                    warning_signs TEXT, coping_strategies TEXT,
                    social_contacts TEXT, professionals TEXT,
                    env_safety TEXT, reasons_to_live TEXT
                );
            """)

    def log_mood(self, score, emotion, note, energy, sleep):
        today = date.today().isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("INSERT INTO mood_journal(date,mood_score,emotion,note,energy,sleep_hours)"
                      " VALUES(?,?,?,?,?,?)", (today,score,emotion,note,energy,sleep))
        self._update_streak()

    def get_mood_history(self, days=14):
        since = (date.today()-timedelta(days=days)).isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            rows = c.execute("SELECT date,mood_score,emotion,note,energy,sleep_hours "
                             "FROM mood_journal WHERE date>=? ORDER BY date", (since,)).fetchall()
        return [{"date":r[0],"score":r[1],"emotion":r[2],"note":r[3],"energy":r[4],"sleep":r[5]} for r in rows]

    def log_gratitude(self, entries):
        today = date.today().isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            for e in entries:
                if e.strip():
                    c.execute("INSERT INTO gratitude(date,entry) VALUES(?,?)", (today,e.strip()))

    def get_gratitude(self, days=30):
        since = (date.today()-timedelta(days=days)).isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            rows = c.execute("SELECT date,entry FROM gratitude WHERE date>=? ORDER BY date DESC",(since,)).fetchall()
        return [{"date":r[0],"entry":r[1]} for r in rows]

    def add_goal(self, title, category):
        with sqlite3.connect(Config.DB_PATH) as c:
            cur = c.execute("INSERT INTO goals(created,title,category) VALUES(?,?,?)",
                            (datetime.now().isoformat(), title, category))
            return cur.lastrowid

    def complete_goal(self, goal_id):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("UPDATE goals SET completed=1, completed_date=? WHERE id=?",
                      (date.today().isoformat(), goal_id))

    def delete_goal(self, goal_id):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("DELETE FROM goals WHERE id=?", (goal_id,))

    def get_goals(self, completed=None):
        with sqlite3.connect(Config.DB_PATH) as c:
            if completed is None:
                rows = c.execute("SELECT id,created,title,category,completed,completed_date "
                                 "FROM goals ORDER BY completed,created DESC").fetchall()
            else:
                rows = c.execute("SELECT id,created,title,category,completed,completed_date "
                                 "FROM goals WHERE completed=? ORDER BY created DESC",
                                 (1 if completed else 0,)).fetchall()
        return [{"id":r[0],"created":r[1],"title":r[2],"category":r[3],
                 "completed":r[4],"completed_date":r[5]} for r in rows]

    def _update_streak(self):
        today     = date.today().isoformat()
        yesterday = (date.today()-timedelta(days=1)).isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            row = c.execute("SELECT last_checkin,current_streak,longest_streak,total_days "
                            "FROM streaks WHERE id=1").fetchone()
            if not row:
                c.execute("INSERT INTO streaks VALUES(1,?,1,1,1)", (today,)); return
            last,cur,longest,total = row
            if last == today: return
            cur = cur+1 if last==yesterday else 1
            c.execute("UPDATE streaks SET last_checkin=?,current_streak=?,longest_streak=?,total_days=?"
                      " WHERE id=1", (today,cur,max(longest,cur),total+1))

    def get_streak(self):
        with sqlite3.connect(Config.DB_PATH) as c:
            row = c.execute("SELECT last_checkin,current_streak,longest_streak,total_days "
                            "FROM streaks WHERE id=1").fetchone()
        return {"current":row[1],"longest":row[2],"total":row[3],"last":row[0]} if row \
               else {"current":0,"longest":0,"total":0,"last":None}

    def log_turn(self, role, content, session_id):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("INSERT INTO conversations(timestamp,role,content,session_id) VALUES(?,?,?,?)",
                      (datetime.now().isoformat(), role, content, session_id))

    def save_assessment(self, tool, score, interpretation, subscores, profile):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("INSERT INTO assessments(date,tool,score,interpretation,subscores,profile) "
                      "VALUES(?,?,?,?,?,?)",
                      (date.today().isoformat(), tool, score,
                       interpretation, str(subscores), profile))

    def get_assessments(self, tool=None, limit=10):
        with sqlite3.connect(Config.DB_PATH) as c:
            if tool:
                rows = c.execute("SELECT date,tool,score,interpretation FROM assessments "
                                 "WHERE tool=? ORDER BY date DESC LIMIT ?", (tool, limit)).fetchall()
            else:
                rows = c.execute("SELECT date,tool,score,interpretation FROM assessments "
                                 "ORDER BY date DESC LIMIT ?", (limit,)).fetchall()
        return [{"date":r[0],"tool":r[1],"score":r[2],"interpretation":r[3]} for r in rows]

    def save_safety_plan(self, **fields):
        now = datetime.now().isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            existing = c.execute("SELECT id FROM safety_plans ORDER BY id DESC LIMIT 1").fetchone()
            if existing:
                c.execute("UPDATE safety_plans SET updated=?,warning_signs=?,coping_strategies=?,"
                          "social_contacts=?,professionals=?,env_safety=?,reasons_to_live=? WHERE id=?",
                          (now, fields.get("warning_signs",""), fields.get("coping_strategies",""),
                           fields.get("social_contacts",""), fields.get("professionals",""),
                           fields.get("env_safety",""), fields.get("reasons_to_live",""),
                           existing[0]))
            else:
                c.execute("INSERT INTO safety_plans(created,updated,warning_signs,coping_strategies,"
                          "social_contacts,professionals,env_safety,reasons_to_live) VALUES(?,?,?,?,?,?,?,?)",
                          (now, now, fields.get("warning_signs",""), fields.get("coping_strategies",""),
                           fields.get("social_contacts",""), fields.get("professionals",""),
                           fields.get("env_safety",""), fields.get("reasons_to_live","")))

    def get_safety_plan(self):
        with sqlite3.connect(Config.DB_PATH) as c:
            row = c.execute("SELECT warning_signs,coping_strategies,social_contacts,"
                            "professionals,env_safety,reasons_to_live,updated "
                            "FROM safety_plans ORDER BY id DESC LIMIT 1").fetchone()
        if not row: return None
        return {"warning_signs":row[0],"coping_strategies":row[1],"social_contacts":row[2],
                "professionals":row[3],"env_safety":row[4],"reasons_to_live":row[5],"updated":row[6]}

    def get_mood_stats(self, days=7):
        since = (date.today()-timedelta(days=days)).isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            rows = c.execute("SELECT mood_score,emotion FROM mood_journal WHERE date>=?", (since,)).fetchall()
        if not rows: return {"avg_score":0,"top_emotion":"—","count":0}
        scores  = [r[0] for r in rows if r[0]]
        emotions= [r[1] for r in rows if r[1]]
        top = Counter(emotions).most_common(1)[0][0] if emotions else "—"
        return {"avg_score":round(sum(scores)/len(scores),1) if scores else 0, "top_emotion":top, "count":len(rows)}


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAIN
# ═══════════════════════════════════════════════════════════════════════════════

class Brain:
    EMOTION_KW = {
        "anxious":     ["anxious","nervous","worried","panic","scared","stressed","fear","dread","anxiety"],
        "sad":         ["sad","depressed","down","hopeless","empty","lonely","miserable","grief","crying"],
        "angry":       ["angry","frustrated","furious","annoyed","mad","irritated","rage","bitter"],
        "overwhelmed": ["overwhelmed","drowning","too much","can't handle","crushing","swamped"],
        "exhausted":   ["exhausted","burnt out","tired","drained","no energy","burned out","depleted"],
        "confused":    ["confused","lost","don't know","unsure","uncertain","unclear"],
        "hopeful":     ["hopeful","better","improving","good","happy","excited","optimistic"],
    }
    CRISIS_KW = ["suicide","kill myself","end my life","want to die","end it all",
                 "take my life","not worth living","better off dead","self harm","hurt myself"]

    def analyze(self, text):
        t = text.lower()
        scores = {}
        for em,kws in self.EMOTION_KW.items():
            c = sum(1 for kw in kws if kw in t)
            if c: scores[em] = min(100, c*28)
        primary = max(scores, key=scores.get) if scores else "neutral"
        is_crisis = any(kw in t for kw in self.CRISIS_KW)
        return {"primary_emotion": primary, "emotion_intensity": scores.get(primary, 0),
                "is_crisis": is_crisis}


# ═══════════════════════════════════════════════════════════════════════════════
#  GROQ
# ═══════════════════════════════════════════════════════════════════════════════

BASE_SYSTEM = """You are Mindful Pro, a warm, emotionally intelligent mental health companion.

Core approach:
- Validate feelings BEFORE any advice or techniques
- Warm and human — use contractions, varied sentence length, never robotic
- Give specific, actionable guidance — never vague platitudes
- Explain briefly WHY a technique works when you suggest it
- End with a gentle question or open space — don't close things off
- 2-3 paragraphs (~150-200 words)
- NEVER diagnose, prescribe, or claim to replace professional therapy
- Crisis language → immediately provide crisis resources"""

def groq_chat(messages, api_key):
    try:
        resp = requests.post(
            Config.GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": Config.GROQ_MODEL, "messages": messages,
                  "temperature": Config.TEMPERATURE, "max_tokens": Config.MAX_TOKENS},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        err = ""
        try: err = resp.json().get("error",{}).get("message","")
        except: pass
        return f"⚠️ Error ({resp.status_code}): {err or resp.text[:100]}"
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except Exception as e:
        return f"⚠️ Error: {e}"

def build_messages(history, user_msg, analysis, profile_key):
    profile = PROFILES.get(profile_key, PROFILES["general"])
    system = BASE_SYSTEM + profile["system_prompt_addon"]
    hint = (f"\n\n[Context: emotion={analysis['primary_emotion']}, "
            f"intensity={analysis['emotion_intensity']}%. Validate first.]")
    msgs = [{"role": "system", "content": system + hint}]
    msgs += history
    msgs.append({"role": "user", "content": user_msg})
    return msgs


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "view":         "profile_select" if not os.path.exists("mindful_pro.db") else "home",
        "profile":      "general",
        "chat_history": [],
        "session_id":   datetime.now().strftime("%Y%m%d%H%M%S"),
        "api_key":      Config.GROQ_API_KEY,
        "affirmation":  random.choice(Config.AFFIRMATIONS),
        "brain":        Brain(),
        "active_tool":  None,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Always start at profile_select if profile not in PROFILES
    if st.session_state.profile not in PROFILES:
        st.session_state.profile = "general"


def P():
    """Current profile dict."""
    return PROFILES.get(st.session_state.profile, PROFILES["general"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TOP BAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_topbar():
    profile = P()
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    with col1:
        if st.session_state.view == "home":
            st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding-top:4px">'
                        f'<span style="font-family:Lora,serif;font-size:1.1rem">🌿 Mindful Pro</span>'
                        f'<span class="profile-badge">{profile["icon"]} {profile["label"]}</span>'
                        f'</div>', unsafe_allow_html=True)
        else:
            if st.button("← Back", key="back_btn"):
                st.session_state.view = "home"
                st.rerun()
    with col2:
        if st.button("🆘", key="sos_btn", help="Crisis resources"):
            st.session_state.view = "crisis"
            st.rerun()
    with col3:
        if st.button("👤", key="profile_btn", help="Change profile"):
            st.session_state.view = "profile_select"
            st.rerun()
    with col4:
        if st.button("⚙", key="settings_btn", help="Settings"):
            st.session_state.view = "settings"
            st.rerun()
    st.markdown("<hr style='margin:10px 0 18px 0'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PROFILE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def page_profile_select():
    st.markdown("## Who are you?")
    st.markdown('<div style="color:#7a8aa0;font-size:0.88rem;margin-bottom:20px">'
                'This shapes the language, tools, and resources you see. '
                'You can change it anytime.</div>', unsafe_allow_html=True)

    for key, info in PROFILES.items():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {info['color']};margin:5px 0;padding:12px 18px">
                <div style="font-size:1.05rem">{info['icon']} <b>{info['label']}</b></div>
                <div style="font-size:0.82rem;color:#7a8aa0;margin-top:2px">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            if st.button("Select", key=f"prof_{key}"):
                st.session_state.profile = key
                st.session_state.view    = "home"
                st.session_state.affirmation = random.choice(Config.AFFIRMATIONS)
                st.rerun()

    st.markdown("<div style='height:16px;color:#4a5568;font-size:0.78rem;text-align:center'>"
                "Your choice stays on this device only.</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HOME — Four entry points
# ═══════════════════════════════════════════════════════════════════════════════

def page_home(db: Database):
    profile = P()
    hour = datetime.now().hour
    if hour < 12:   greeting = "Good morning."
    elif hour < 17: greeting = "Good afternoon."
    else:           greeting = "Good evening."

    st.markdown(f"## {greeting}{profile['language']['greeting_suffix']}")
    st.markdown(f'<div class="affirmation">"{st.session_state.affirmation}"</div>',
                unsafe_allow_html=True)

    st.markdown("**What brings you here today?**")
    st.markdown('<div style="color:#7a8aa0;font-size:0.86rem;margin-bottom:16px">'
                'Choose what feels most true right now.</div>', unsafe_allow_html=True)

    entries = [
        ("right_now",  "🌊", f"I need help right now",   f"Something is overwhelming me"),
        ("talk",       "💬", "I want to talk",            profile["language"]["talk_intro"][:50] + "…"),
        ("reflect",    "🪞", "I want to reflect",         "Mood, gratitude, patterns"),
        ("build",      "🌱", "I want to build habits",    "Goals, tracking, progress"),
        ("assess",     "📋", "I want to assess myself",   "Validated screening tools (PHQ-9, GAD-7, etc.)"),
        ("learn",      "📖", "I want to understand",      "Psychoeducation library"),
        ("safety",     "🛡",  "Safety planning",           "Build or review your personal safety plan"),
    ]

    for view, icon, label, desc in entries:
        if st.button(f"{icon}  {label}", key=f"nav_{view}", use_container_width=True,
                     help=desc):
            st.session_state.view = view
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-size:0.76rem;color:#3a4558">'
                'Crisis? Tap 🆘 above for immediate support lines.</div>', unsafe_allow_html=True)

    # Recent check-ins — soft, below the fold
    history = db.get_mood_history(7)
    if history:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div style="color:#7a8aa0;font-size:0.8rem;margin-bottom:8px">This week</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(history[-7:]))
        for i, entry in enumerate(history[-7:]):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align:center;padding:6px 2px">
                    <div style="font-size:1.1rem">{_em_emoji(entry.get('emotion',''))}</div>
                    <div style="font-size:0.65rem;color:#7a8aa0">{entry['date'][-5:]}</div>
                    <div style="font-size:0.75rem;color:#a0aec0">{entry.get('score') or '—'}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  RIGHT NOW — Immediate distress & grounding
# ═══════════════════════════════════════════════════════════════════════════════

def page_right_now():
    profile = P()
    st.markdown("## I'm here with you.")
    st.markdown(f'<div style="color:#7a8aa0;margin-bottom:16px">'
                f'Let\'s slow things down. Choose whatever calls to you.</div>',
                unsafe_allow_html=True)

    with st.expander("🆘 Crisis lines — tap to expand"):
        render_crisis_lines()

    tools = [
        ("box",       "📦 Box breathing",           "4-4-4-4 — used by military & first responders"),
        ("478",       "🫁 4-7-8 breathing",          "Activates your calm-down response"),
        ("grounding", "🧘 5-4-3-2-1 Grounding",      "Anchors you to the present moment"),
        ("scan",      "🚶 Body scan",                 "Notice without judgment"),
        ("cold",      "🧊 Cold-water reset",          "Physiological sigh + cold water — fast acting"),
        ("tipp",      "🏃 TIPP skills (DBT)",         "Temperature · Intense exercise · Paced breathing · Paired relaxation"),
        ("soothe",    "🌊 Self-soothing",             "Five senses grounding — slower, deeper"),
    ]

    for key, label, desc in tools:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"""
            <div class="card card-blue" style="padding:10px 16px;margin:4px 0">
                <span style="font-weight:500">{label}</span>
                <div style="font-size:0.79rem;color:#7a8aa0;margin-top:2px">{desc}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            if st.button("Start", key=f"rn_{key}"):
                st.session_state.active_tool = key
                st.rerun()

    tool = st.session_state.active_tool
    if tool:
        st.markdown("---")
        if tool == "box":        _tool_box()
        elif tool == "478":      _tool_478()
        elif tool == "grounding":_tool_grounding()
        elif tool == "scan":     _tool_body_scan()
        elif tool == "cold":     _tool_cold_reset()
        elif tool == "tipp":     _tool_tipp()
        elif tool == "soothe":   _tool_soothe()
        if st.button("✖ Close tool"):
            st.session_state.active_tool = None
            st.rerun()


def _tool_box():
    st.markdown("### 📦 Box Breathing")
    st.markdown("""<div class="card card-blue">
    <p style="line-height:2.2;margin:0;font-size:0.95rem">
        <span style="color:#6fa8dc">① IN</span> — 4 counts<br>
        <span style="color:#7ec8a0">② HOLD</span> — 4 counts<br>
        <span style="color:#6fa8dc">③ OUT</span> — 4 counts<br>
        <span style="color:#7ec8a0">④ HOLD</span> — 4 counts
    </p>
    <div style="margin-top:12px;font-size:0.82rem;color:#7a8aa0">
        Repeat 4–6 cycles. Used by Navy SEALs and first responders to regulate under pressure.
        The symmetrical pattern resets the autonomic nervous system — no equipment needed.
    </div></div>""", unsafe_allow_html=True)

def _tool_478():
    st.markdown("### 🫁 4-7-8 Breathing")
    st.markdown("""<div class="card card-blue">
    <p style="line-height:2.2;margin:0">
        <span style="color:#6fa8dc">① Breathe IN</span> — 4 counts (through nose)<br>
        <span style="color:#9b8ec4">② HOLD</span> — 7 counts<br>
        <span style="color:#7ec8a0">③ Breathe OUT</span> — 8 counts (through mouth, audible)
    </p>
    <div style="margin-top:12px;font-size:0.82rem;color:#7a8aa0">
        The extended exhale stimulates the vagus nerve — your body's main calming pathway.
        Even 2 rounds measurably reduces heart rate. Do 3–4 cycles. If lightheaded, breathe normally first.
    </div></div>""", unsafe_allow_html=True)

def _tool_grounding():
    st.markdown("### 🧘 5-4-3-2-1 Grounding")
    st.markdown('<div style="color:#7a8aa0;font-size:0.85rem;margin-bottom:10px">Anxiety lives in the future. This pulls you back to now.</div>', unsafe_allow_html=True)
    for sense, label, hint in [
        ("5","👁 5 things you SEE",   "Small details — a crack in the ceiling, dust on a shelf"),
        ("4","✋ 4 things you TOUCH", "Textures, temperatures, weight"),
        ("3","👂 3 things you HEAR",  "Near, distant, internal (your own breath)"),
        ("2","👃 2 things you SMELL", "Subtle — air, skin, the room"),
        ("1","👅 1 thing you TASTE",  "Whatever's present right now"),
    ]:
        with st.expander(label):
            st.caption(hint)
            for i in range(int(sense)):
                st.text_input(f"  {i+1}.", key=f"g_{sense}_{i}", placeholder="…")
    st.success("✨ You made it to this moment.")

def _tool_body_scan():
    st.markdown("### 🚶 Body Scan")
    st.markdown("""<div class="card card-blue" style="font-size:0.9rem;line-height:2">
    Soften your gaze or close your eyes.<br><br>
    Start at the crown of your head — just <em>notice</em>, without fixing.<br>
    Move slowly: <b>forehead → jaw → neck → shoulders → chest → belly → lower back → hands → legs → feet.</b><br><br>
    Where you notice tension: breathe into it. Say internally: <em>"I notice this. It's okay."</em><br><br>
    You don't have to release anything. Noticing is enough.
    </div>""", unsafe_allow_html=True)

def _tool_cold_reset():
    st.markdown("### 🧊 Cold-Water Reset")
    st.markdown("""<div class="card card-blue" style="font-size:0.9rem;line-height:1.9">
    <b>Physiological sigh</b> — fastest known acute stress reducer:<br>
    • Double inhale through the nose (short, then longer)<br>
    • One long exhale through the mouth<br>
    • Repeat 1–3 times<br><br>
    <b>Cold water</b> (dive reflex activation):<br>
    • Splash cold water on your face or wrists<br>
    • Hold wrists under cold running water for 30 seconds<br>
    • Activates the mammalian dive reflex — slows heart rate within seconds<br><br>
    <div style="font-size:0.8rem;color:#7a8aa0">
    This is what DBT calls a TIPP Temperature intervention. Effective for high-intensity distress.
    </div></div>""", unsafe_allow_html=True)

def _tool_tipp():
    st.markdown("### 🏃 TIPP Skills (DBT)")
    st.markdown('<div style="color:#7a8aa0;font-size:0.84rem;margin-bottom:10px">Dialectical Behavior Therapy skills for intense emotional distress. Work on the body first — the mind follows.</div>', unsafe_allow_html=True)
    skills = [
        ("T — Temperature", "Cold water on face / wrists / ice pack. Activates dive reflex. Fastest physiological down-regulation available."),
        ("I — Intense Exercise", "60 seconds of intense movement (jumping jacks, sprints, push-ups). Burns off stress hormones — adrenaline and cortisol metabolize through movement."),
        ("P — Paced Breathing", "Exhale longer than inhale. Try 4 in, 6 out. Or simply make your exhale twice as long as your inhale. Extended exhale = parasympathetic activation."),
        ("P — Paired Muscle Relaxation", "Inhale + tense a muscle group. Exhale + release. Work through the body. The contrast between tension and release deepens relaxation."),
    ]
    for title, desc in skills:
        st.markdown(f"""<div class="card card-lav" style="margin:6px 0;padding:12px 16px">
        <div style="font-weight:500;font-size:0.95rem">{title}</div>
        <div style="font-size:0.82rem;color:#7a8aa0;margin-top:5px;line-height:1.6">{desc}</div>
        </div>""", unsafe_allow_html=True)

def _tool_soothe():
    st.markdown("### 🌊 Self-Soothing with Five Senses")
    st.markdown('<div style="color:#7a8aa0;font-size:0.84rem;margin-bottom:10px">Slower than grounding — for gentler distress. Find one thing to savor in each sense.</div>', unsafe_allow_html=True)
    senses = [
        ("👁 Vision",   "Find something beautiful near you. Really look at it — color, shape, light."),
        ("👂 Hearing",  "Find a sound you find calming. Or create silence and listen to it."),
        ("👃 Smell",    "Coffee, a candle, fresh air, soap, clothing. Smell is the fastest sense to shift state."),
        ("✋ Touch",    "Fabric, temperature, texture. Hold something comforting."),
        ("👅 Taste",    "Warm drink, something sweet or savory. Eat or drink slowly, noticing flavor."),
    ]
    for sense, prompt in senses:
        st.markdown(f"""<div class="card" style="padding:10px 16px;margin:5px 0">
        <div style="font-weight:500">{sense}</div>
        <div style="font-size:0.82rem;color:#7a8aa0;margin-top:3px">{prompt}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TALK — AI companion
# ═══════════════════════════════════════════════════════════════════════════════

def page_talk(db: Database):
    profile = P()
    st.markdown("## I'm listening.")
    st.markdown(f'<div style="color:#7a8aa0;font-size:0.86rem;margin-bottom:14px">'
                f'{profile["language"]["talk_intro"]}</div>', unsafe_allow_html=True)

    if not st.session_state.api_key:
        st.markdown("""<div class="card card-amber" style="padding:18px">
        <div style="font-size:0.93rem;margin-bottom:8px">To use the AI companion, you need a free Groq API key.</div>
        <div style="font-size:0.83rem;color:#7a8aa0">Go to ⚙ Settings above. Get a free key at
        <a href="https://console.groq.com" target="_blank" style="color:#6fa8dc">console.groq.com</a>
        — takes about a minute.</div></div>""", unsafe_allow_html=True)
        return

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div><div class="msg-label" style="text-align:right">You</div>'
                        f'<div class="msg-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div><div class="msg-label msg-label-bot">{P()["icon"]} Mindful Pro</div>'
                        f'<div class="msg-bot">{msg["content"].replace(chr(10),"<br>")}</div></div>',
                        unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown('<div style="text-align:center;color:#3a4558;font-size:0.86rem;padding:28px 0 8px 0">'
                    'There\'s no right way to start.</div>', unsafe_allow_html=True)

    user_input = st.chat_input("What's on your mind?")
    if user_input:
        analysis = st.session_state.brain.analyze(user_input)
        if analysis["is_crisis"]:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            render_crisis_lines()
            reply = ("I'm really glad you said something. Right now, your safety is the only thing that matters. "
                     "Please reach out to one of the numbers above — trained people are available right now, 24/7. "
                     "What you're feeling won't last forever. You don't have to face this alone. 💙")
            st.session_state.chat_history.append({"role":"assistant","content":reply})
            db.log_turn("user", user_input, st.session_state.session_id)
            db.log_turn("assistant", reply, st.session_state.session_id)
            st.rerun()
            return

        messages = build_messages(st.session_state.chat_history, user_input,
                                  analysis, st.session_state.profile)
        with st.spinner(""):
            reply = groq_chat(messages, st.session_state.api_key)

        st.session_state.chat_history.append({"role":"user","content":user_input})
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        db.log_turn("user", user_input, st.session_state.session_id)
        db.log_turn("assistant", reply, st.session_state.session_id)
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Start new conversation"):
            st.session_state.chat_history = []
            st.session_state.session_id   = datetime.now().strftime("%Y%m%d%H%M%S")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATED ASSESSMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def page_assess(db: Database):
    profile = P()
    available = profile["assessments"]

    st.markdown("## Clinical Screening Tools")
    st.markdown(f"""<div class="card card-amber" style="font-size:0.82rem;color:#c8a060;padding:12px 16px;margin-bottom:16px">
    ⚠️ These are validated <b>screening tools</b>, not diagnostic instruments. Scores indicate whether further
    evaluation is warranted — not a diagnosis. Always consult a qualified professional for interpretation.
    </div>""", unsafe_allow_html=True)

    tool_tabs = {k: k for k in available}
    tabs = st.tabs(list(available))

    for i, tool in enumerate(available):
        with tabs[i]:
            if tool == "PHQ-9":   _assess_phq9(db)
            elif tool == "GAD-7": _assess_gad7(db)
            elif tool == "PSS-4": _assess_pss4(db)
            elif tool == "PCL-5": _assess_pcl5(db)
            elif tool == "ProQOL":_assess_proqol(db)
            elif tool == "MBI":   _assess_mbi(db)


def _score_bar(score, max_score, color="#6fa8dc"):
    pct = int(score / max_score * 100)
    return (f'<div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%;'
            f'background:{color}"></div></div>')

def _assess_phq9(db):
    st.markdown("### PHQ-9 — Patient Health Questionnaire")
    st.markdown('<div style="color:#7a8aa0;font-size:0.82rem;margin-bottom:12px">'
                'Over the <b>last 2 weeks</b>, how often have you been bothered by the following?</div>',
                unsafe_allow_html=True)
    scores = []
    with st.form("phq9"):
        for i, item in enumerate(PHQ9_ITEMS):
            val = st.selectbox(f"{i+1}. {item}", PHQ9_OPTIONS, key=f"phq9_{i}")
            scores.append(PHQ9_OPTIONS.index(val))
            if i == 8 and val != "Not at all":
                st.warning("⚠️ Item 9 asks about thoughts of self-harm. If you're having these thoughts, please reach out for support — see 🆘 above.")
        submitted = st.form_submit_button("Calculate score", use_container_width=True)

    if submitted:
        total = sum(scores)
        if total <= 4:   interp, col = "Minimal depression", "#7ec8a0"
        elif total <= 9: interp, col = "Mild depression", "#d4a843"
        elif total <= 14:interp, col = "Moderate depression", "#c4906e"
        elif total <= 19:interp, col = "Moderately severe depression", "#c97b84"
        else:            interp, col = "Severe depression", "#c97b84"

        st.markdown(f"""<div class="card" style="border-left:3px solid {col};margin-top:12px">
        <div style="font-family:Lora,serif;font-size:2rem;color:{col}">{total}/27</div>
        <div style="font-size:0.9rem;margin-top:4px">{interp}</div>
        {_score_bar(total,27,col)}
        <div style="font-size:0.78rem;color:#7a8aa0;margin-top:8px">
        Scores ≥10 suggest clinical significance. ≥20 typically indicates severe depression
        requiring prompt professional evaluation. This is a screening tool, not a diagnosis.
        </div></div>""", unsafe_allow_html=True)
        db.save_assessment("PHQ-9", total, interp, {"items": scores}, st.session_state.profile)
        _show_history(db, "PHQ-9")

def _assess_gad7(db):
    st.markdown("### GAD-7 — Generalized Anxiety Disorder Scale")
    st.markdown('<div style="color:#7a8aa0;font-size:0.82rem;margin-bottom:12px">'
                'Over the <b>last 2 weeks</b>, how often have you been bothered by:</div>',
                unsafe_allow_html=True)
    scores = []
    with st.form("gad7"):
        for i, item in enumerate(GAD7_ITEMS):
            val = st.selectbox(f"{i+1}. {item}", PHQ9_OPTIONS, key=f"gad7_{i}")
            scores.append(PHQ9_OPTIONS.index(val))
        submitted = st.form_submit_button("Calculate score", use_container_width=True)

    if submitted:
        total = sum(scores)
        if total <= 4:    interp, col = "Minimal anxiety", "#7ec8a0"
        elif total <= 9:  interp, col = "Mild anxiety", "#d4a843"
        elif total <= 14: interp, col = "Moderate anxiety", "#c4906e"
        else:             interp, col = "Severe anxiety", "#c97b84"

        st.markdown(f"""<div class="card" style="border-left:3px solid {col};margin-top:12px">
        <div style="font-family:Lora,serif;font-size:2rem;color:{col}">{total}/21</div>
        <div style="font-size:0.9rem;margin-top:4px">{interp}</div>
        {_score_bar(total,21,col)}
        <div style="font-size:0.78rem;color:#7a8aa0;margin-top:8px">
        Score ≥10 suggests clinically significant anxiety. GAD-7 also screens for panic disorder,
        social anxiety, and PTSD. Further evaluation recommended at ≥8.
        </div></div>""", unsafe_allow_html=True)
        db.save_assessment("GAD-7", total, interp, {"items": scores}, st.session_state.profile)
        _show_history(db, "GAD-7")

def _assess_pss4(db):
    st.markdown("### PSS-4 — Perceived Stress Scale")
    st.markdown('<div style="color:#7a8aa0;font-size:0.82rem;margin-bottom:12px">'
                'In the <b>last month</b>, how often have you felt...</div>',
                unsafe_allow_html=True)
    scores = []
    with st.form("pss4"):
        for i, item in enumerate(PSS4_ITEMS):
            val = st.selectbox(f"{i+1}. {item}", PSS4_OPTIONS, key=f"pss4_{i}")
            raw = PSS4_OPTIONS.index(val)
            # Items 3 and 4 are reverse scored
            scores.append(4 - raw if i >= 2 else raw)
        submitted = st.form_submit_button("Calculate score", use_container_width=True)

    if submitted:
        total = sum(scores)
        if total <= 4:    interp, col = "Low perceived stress", "#7ec8a0"
        elif total <= 8:  interp, col = "Moderate perceived stress", "#d4a843"
        else:             interp, col = "High perceived stress", "#c97b84"

        st.markdown(f"""<div class="card" style="border-left:3px solid {col};margin-top:12px">
        <div style="font-family:Lora,serif;font-size:2rem;color:{col}">{total}/16</div>
        <div style="font-size:0.9rem;margin-top:4px">{interp}</div>
        {_score_bar(total,16,col)}
        <div style="font-size:0.78rem;color:#7a8aa0;margin-top:8px">
        PSS-4 measures perceived helplessness and self-efficacy under stress. Items 3 and 4
        are reverse scored. High scores correlate with health outcomes, burnout, and immune function.
        </div></div>""", unsafe_allow_html=True)
        db.save_assessment("PSS-4", total, interp, {"items": scores}, st.session_state.profile)
        _show_history(db, "PSS-4")

def _assess_pcl5(db):
    st.markdown("### PCL-5 — PTSD Checklist")
    st.markdown(f"""<div class="card card-amber" style="font-size:0.82rem;color:#c8a060;padding:10px 16px;margin-bottom:12px">
    This tool screens for PTSD symptoms related to a stressful experience. Completing it may bring up difficult memories.
    Crisis lines are accessible via 🆘 above. This is a screening tool only — clinical diagnosis requires professional evaluation.
    </div>""", unsafe_allow_html=True)
    st.markdown('<div style="color:#7a8aa0;font-size:0.82rem;margin-bottom:12px">'
                'In the past month, how much were you bothered by:</div>', unsafe_allow_html=True)
    scores = []
    with st.form("pcl5"):
        for i, item in enumerate(PCL5_ITEMS):
            val = st.selectbox(f"{i+1}. {item}", PCL5_OPTIONS, key=f"pcl5_{i}")
            scores.append(PCL5_OPTIONS.index(val))
        submitted = st.form_submit_button("Calculate score", use_container_width=True)

    if submitted:
        total = sum(scores)
        # DSM-5 cluster scores
        b = sum(scores[0:5])   # Intrusion
        c = sum(scores[5:7])   # Avoidance
        d = sum(scores[7:14])  # Negative alterations
        e = sum(scores[14:20]) # Arousal

        if total >= 33:   interp, col = "Probable PTSD — professional evaluation strongly recommended", "#c97b84"
        elif total >= 20: interp, col = "Subclinical PTSD symptoms — professional evaluation recommended", "#c4906e"
        else:             interp, col = "Below PTSD threshold at this time", "#7ec8a0"

        st.markdown(f"""<div class="card" style="border-left:3px solid {col};margin-top:12px">
        <div style="font-family:Lora,serif;font-size:2rem;color:{col}">{total}/80</div>
        <div style="font-size:0.9rem;margin-top:4px">{interp}</div>
        {_score_bar(total,80,col)}
        <div style="margin-top:12px;font-size:0.83rem">
            <div style="color:#7a8aa0;margin-bottom:6px">Cluster scores:</div>
            <div>Intrusion (B): {b}/20 · Avoidance (C): {c}/8 · Neg. cognition (D): {d}/28 · Arousal (E): {e}/24</div>
        </div>
        <div style="font-size:0.78rem;color:#7a8aa0;margin-top:10px">
        Cutpoint of 33 is recommended for probable PTSD in military/veteran populations. Civilian cutpoint is often 31.
        PCL-5 is also used to monitor treatment progress over time.
        </div></div>""", unsafe_allow_html=True)
        db.save_assessment("PCL-5", total, interp, {"B":b,"C":c,"D":d,"E":e}, st.session_state.profile)
        _show_history(db, "PCL-5")

def _assess_proqol(db):
    st.markdown("### ProQOL — Professional Quality of Life")
    st.markdown('<div style="color:#7a8aa0;font-size:0.82rem;margin-bottom:12px">'
                'Based on your work helping others in the <b>last 30 days</b>:</div>',
                unsafe_allow_html=True)
    cs_scores, bo_scores, sts_scores = [], [], []
    with st.form("proqol"):
        for i, (subscale, item) in enumerate(PROQOL_ITEMS):
            val = st.selectbox(f"{i+1}. {item}", PROQOL_OPTIONS, key=f"proqol_{i}")
            score = PROQOL_OPTIONS.index(val)
            if subscale == "cs":  cs_scores.append(score)
            elif subscale == "bo": bo_scores.append(score)
            else:                  sts_scores.append(score)
        submitted = st.form_submit_button("Calculate", use_container_width=True)

    if submitted:
        cs  = sum(cs_scores)
        bo  = sum(bo_scores)
        sts = sum(sts_scores)

        def level(s, max_s=16):
            pct = s/max_s
            if pct >= 0.75: return "High", "#c97b84"
            elif pct >= 0.5: return "Moderate", "#d4a843"
            else:           return "Low", "#7ec8a0"

        cs_l, cs_c   = level(cs)
        bo_l, bo_c   = level(bo)
        sts_l, sts_c = level(sts)

        st.markdown(f"""<div class="card card-lav" style="margin-top:12px">
        <div style="font-size:0.9rem;font-weight:500;margin-bottom:12px">ProQOL Results</div>
        <div style="margin:8px 0">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                <span>Compassion Satisfaction</span>
                <span style="color:{cs_c}">{cs_l} ({cs}/16)</span>
            </div>
            {_score_bar(cs,16,cs_c)}
            <div style="font-size:0.75rem;color:#7a8aa0">High = drawing meaning from your work. Low may indicate burnout risk.</div>
        </div>
        <div style="margin:8px 0">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                <span>Burnout</span>
                <span style="color:{bo_c}">{bo_l} ({bo}/16)</span>
            </div>
            {_score_bar(bo,16,bo_c)}
            <div style="font-size:0.75rem;color:#7a8aa0">High = chronic depletion, overwhelm, hopelessness about your work.</div>
        </div>
        <div style="margin:8px 0">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                <span>Secondary Traumatic Stress</span>
                <span style="color:{sts_c}">{sts_l} ({sts}/16)</span>
            </div>
            {_score_bar(sts,16,sts_c)}
            <div style="font-size:0.75rem;color:#7a8aa0">High = trauma symptoms related to others' experiences. May indicate vicarious traumatization.</div>
        </div>
        <div style="font-size:0.75rem;color:#7a8aa0;margin-top:10px;padding-top:10px;border-top:1px solid #252f45">
        Supervision, consultation, and your own therapy are the primary protective factors against
        burnout and secondary traumatic stress in helping professions.
        </div></div>""", unsafe_allow_html=True)
        db.save_assessment("ProQOL", (cs+bo+sts)/3, f"CS:{cs_l}/BO:{bo_l}/STS:{sts_l}",
                           {"cs":cs,"bo":bo,"sts":sts}, st.session_state.profile)

def _assess_mbi(db):
    st.markdown("### MBI — Maslach Burnout Inventory (Screening)")
    st.markdown('<div style="color:#7a8aa0;font-size:0.82rem;margin-bottom:12px">'
                'How often do you experience the following? (Simplified screening version)</div>',
                unsafe_allow_html=True)
    ee_scores, dp_scores, pa_scores = [], [], []
    with st.form("mbi"):
        for i, (subscale, item) in enumerate(MBI_ITEMS):
            val = st.selectbox(f"{i+1}. {item}", MBI_OPTIONS, key=f"mbi_{i}")
            score = MBI_OPTIONS.index(val)
            if subscale == "ee":  ee_scores.append(score)
            elif subscale == "dp": dp_scores.append(score)
            else:                  pa_scores.append(score)
        submitted = st.form_submit_button("Calculate", use_container_width=True)

    if submitted:
        ee = sum(ee_scores)
        dp = sum(dp_scores)
        pa = sum(pa_scores)  # PA is reverse indicator — low PA = burnout risk

        def ee_level(s):
            return ("High", "#c97b84") if s >= 18 else ("Moderate", "#d4a843") if s >= 10 else ("Low", "#7ec8a0")
        def dp_level(s):
            return ("High", "#c97b84") if s >= 10 else ("Moderate", "#d4a843") if s >= 5 else ("Low", "#7ec8a0")
        def pa_level(s):
            return ("Low", "#c97b84") if s <= 15 else ("Moderate", "#d4a843") if s <= 25 else ("High", "#7ec8a0")

        ee_l, ee_c = ee_level(ee)
        dp_l, dp_c = dp_level(dp)
        pa_l, pa_c = pa_level(pa)

        st.markdown(f"""<div class="card card-teal" style="margin-top:12px">
        <div style="font-size:0.9rem;font-weight:500;margin-bottom:12px">MBI Burnout Profile</div>
        <div style="margin:8px 0">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                <span>Emotional Exhaustion</span>
                <span style="color:{ee_c}">{ee_l} ({ee}/24)</span>
            </div>
            {_score_bar(ee,24,ee_c)}
        </div>
        <div style="margin:8px 0">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                <span>Depersonalization</span>
                <span style="color:{dp_c}">{dp_l} ({dp}/18)</span>
            </div>
            {_score_bar(dp,18,dp_c)}
        </div>
        <div style="margin:8px 0">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem">
                <span>Personal Accomplishment</span>
                <span style="color:{pa_c}">{pa_l} ({pa}/30)</span>
            </div>
            {_score_bar(pa,30,pa_c)}
            <div style="font-size:0.75rem;color:#7a8aa0">Low PA contributes to burnout.</div>
        </div>
        <div style="font-size:0.75rem;color:#7a8aa0;margin-top:10px;padding-top:10px;border-top:1px solid #252f45">
        High EE + High DP + Low PA = burnout profile. Burnout is driven by systemic factors
        (workload, autonomy, community, fairness) — individual self-care alone is insufficient.
        </div></div>""", unsafe_allow_html=True)
        db.save_assessment("MBI", (ee+dp)/2, f"EE:{ee_l}/DP:{dp_l}/PA:{pa_l}",
                           {"ee":ee,"dp":dp,"pa":pa}, st.session_state.profile)

def _show_history(db, tool):
    history = db.get_assessments(tool=tool, limit=5)
    if len(history) > 1:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"**Your {tool} history**")
        for entry in history:
            st.markdown(f'<div style="font-size:0.8rem;color:#7a8aa0;padding:3px 0">'
                        f'{entry["date"]} — <span style="color:#a0b0c0">{entry["score"]:.0f}</span> '
                        f'({entry["interpretation"]})</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PSYCHOEDUCATION LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

def page_learn():
    profile = P()
    available = profile["psychoed_topics"]
    all_topics = list(PSYCHOED.keys())

    st.markdown("## Psychoeducation Library")
    st.markdown('<div style="color:#7a8aa0;font-size:0.86rem;margin-bottom:16px">'
                'Understanding what you\'re experiencing is itself therapeutic.</div>',
                unsafe_allow_html=True)

    # Profile-recommended topics first
    st.markdown("**Recommended for your profile**")
    for topic_key in available:
        if topic_key in PSYCHOED:
            topic = PSYCHOED[topic_key]
            color = _topic_color(topic.get("color","blue"))
            if st.button(f"{topic['title']} — {topic['subtitle']}", key=f"learn_{topic_key}",
                         use_container_width=True):
                st.session_state["reading_topic"] = topic_key
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.expander("📚 Full library"):
        for topic_key, topic in PSYCHOED.items():
            if topic_key not in available:
                if st.button(f"{topic['title']}", key=f"learn_all_{topic_key}",
                             use_container_width=True):
                    st.session_state["reading_topic"] = topic_key
                    st.rerun()

    # Render article
    reading = st.session_state.get("reading_topic")
    if reading and reading in PSYCHOED:
        st.markdown("---")
        topic = PSYCHOED[reading]
        color = _topic_color(topic.get("color", "blue"))
        st.markdown(f'<div style="font-size:0.72rem;color:#7a8aa0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Psychoeducation</div>', unsafe_allow_html=True)
        st.markdown(f"## {topic['title']}")
        st.markdown(f'<div style="color:#7a8aa0;font-style:italic;margin-bottom:18px">{topic["subtitle"]}</div>',
                    unsafe_allow_html=True)
        for section_title, section_body in topic["sections"]:
            st.markdown(f"""<div class="card card-{topic.get('color','blue')}" style="margin:8px 0">
            <div style="font-weight:500;font-size:0.95rem;margin-bottom:6px">{section_title}</div>
            <div style="font-size:0.87rem;color:#c0cce0;line-height:1.75">{section_body}</div>
            </div>""", unsafe_allow_html=True)
        if st.button("← Back to library"):
            del st.session_state["reading_topic"]
            st.rerun()

def _topic_color(name):
    return {"blue":"#6fa8dc","green":"#7ec8a0","amber":"#d4a843","rose":"#c97b84",
            "teal":"#5ba8a0","lavender":"#9b8ec4","orange":"#c4906e"}.get(name,"#6fa8dc")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFETY PLANNING (Stanley-Brown model)
# ═══════════════════════════════════════════════════════════════════════════════

def page_safety(db: Database):
    st.markdown("## Personal Safety Plan")
    st.markdown(f"""<div class="card card-rose" style="font-size:0.83rem;color:#d0a0a8;padding:12px 16px;margin-bottom:16px">
    A safety plan is a prioritized written list of coping strategies and sources of support you can use
    during a crisis. Research shows safety planning significantly reduces suicidal behavior.
    This is your plan — personalize it. Review it when you're well so it's ready when you're not.
    </div>""", unsafe_allow_html=True)

    existing = db.get_safety_plan()
    defaults = existing or {}

    with st.form("safety_plan_form"):
        st.markdown("#### Step 1: Warning signs")
        st.caption("Thoughts, images, feelings, behaviors that tell you a crisis may be developing")
        warning = st.text_area("My warning signs:", value=defaults.get("warning_signs",""),
                               placeholder="e.g. Withdrawing from others, not sleeping, feeling hopeless about the future, thinking 'I can't do this anymore'",
                               height=90)

        st.markdown("#### Step 2: Internal coping strategies")
        st.caption("Things I can do on my own to distract or soothe myself")
        coping = st.text_area("My internal strategies:", value=defaults.get("coping_strategies",""),
                              placeholder="e.g. Box breathing for 5 minutes, go for a walk, listen to specific playlist, do the 5-4-3-2-1 grounding exercise",
                              height=90)

        st.markdown("#### Step 3: Social contacts for distraction")
        st.caption("People and settings that take my mind off things (not necessarily to talk about the crisis)")
        social = st.text_area("People/places:", value=defaults.get("social_contacts",""),
                              placeholder="e.g. Call [name], go to [coffee shop], visit [friend]",
                              height=70)

        st.markdown("#### Step 4: People I can ask for help")
        st.caption("People I trust enough to tell them I'm struggling")
        profs = st.text_area("People who can help:", value=defaults.get("professionals",""),
                             placeholder="e.g. [Name]: [phone]\nTherapist: [name + phone]\nCrisis line: 988",
                             height=90)

        st.markdown("#### Step 5: Making my environment safer")
        st.caption("Reducing access to means during a crisis")
        env = st.text_area("Steps I can take:", value=defaults.get("env_safety",""),
                           placeholder="e.g. Ask [person] to hold my medication, remove [item] from home, lock [item] away",
                           height=70)

        st.markdown("#### Step 6: Reasons for living")
        st.caption("The most important reasons I have to stay alive — in my own words")
        reasons = st.text_area("My reasons:", value=defaults.get("reasons_to_live",""),
                               placeholder="e.g. My family, my dog, the trip I'm planning, the person I want to become…",
                               height=80)

        submitted = st.form_submit_button("💾 Save safety plan", use_container_width=True)

    if submitted:
        db.save_safety_plan(warning_signs=warning, coping_strategies=coping,
                            social_contacts=social, professionals=profs,
                            env_safety=env, reasons_to_live=reasons)
        st.success("✅ Safety plan saved. It's stored only on this device.")

    if existing:
        st.markdown(f'<div style="font-size:0.75rem;color:#4a5568;margin-top:8px">'
                    f'Last updated: {existing.get("updated","")[:10]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Universal crisis resources**")
    for name, number in Config.CRISIS_UNIVERSAL:
        st.markdown(f'<div style="font-size:0.84rem;padding:3px 0">'
                    f'<span style="color:#8fa0b4">{name}</span>'
                    f'<span style="float:right;color:#d4a843">{number}</span></div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  REFLECT
# ═══════════════════════════════════════════════════════════════════════════════

def page_reflect(db: Database):
    st.markdown("## Reflect")
    tab1, tab2, tab3 = st.tabs(["🌡 Mood", "🙏 Gratitude", "📊 Patterns"])
    with tab1: _reflect_mood(db)
    with tab2: _reflect_gratitude(db)
    with tab3: _reflect_patterns(db)

def _reflect_mood(db):
    history = db.get_mood_history(1)
    if history and history[-1]["date"] == date.today().isoformat():
        entry = history[-1]
        st.success("✅ Check-in complete for today")
        c1,c2,c3,c4 = st.columns(4)
        with c1: _metric(str(entry['score'])+"/10","Mood")
        with c2: _metric(_em_emoji(entry.get('emotion','')), entry.get('emotion','').title() or "—")
        with c3: _metric(str(entry['energy'])+"/5","Energy")
        with c4: _metric(str(entry['sleep'])+"h","Sleep")
        if entry.get("note"):
            st.markdown(f'<div class="card" style="margin-top:8px;font-style:italic;color:#8fa0b4">"{entry["note"]}"</div>', unsafe_allow_html=True)
        return

    with st.form("checkin_form"):
        mood = st.slider("How's your mood?", 1, 10, 5)
        labels = {1:"😣 Very dark",2:"😢 Really hard",3:"😔 Low",4:"😐 Flat",5:"🙂 Okay",
                  6:"🙂 Decent",7:"😊 Pretty good",8:"😃 Good",9:"🤩 Really good",10:"✨ Wonderful"}
        st.caption(labels.get(mood,""))
        emotion_opts = ["Anxious","Sad","Angry","Overwhelmed","Exhausted","Confused","Neutral","Hopeful","Happy"]
        emotion = st.selectbox("Dominant emotion?", emotion_opts)
        energy  = st.slider("Energy level", 1, 5, 3)
        sleep   = st.number_input("Hours slept?", 0.0, 24.0, 7.0, 0.5)
        note    = st.text_input("One sentence about today (optional)", placeholder="e.g. Hard morning, better afternoon")
        submitted = st.form_submit_button("✅ Save", use_container_width=True)

    if submitted:
        db.log_mood(mood, emotion.lower(), note, energy, sleep)
        st.balloons()
        if mood <= 3: st.info("💙 Hard days are real. Consider reaching out to someone you trust, or try a grounding exercise.")
        elif sleep < 6: st.warning("😴 Under 6 hours of sleep significantly amplifies emotional reactivity.")
        else: st.success("Saved. Showing up matters.")
        st.rerun()

def _reflect_gratitude(db):
    prompts = random.sample([
        "Something small that made you smile today",
        "Someone who made your life easier recently",
        "A challenge that taught you something",
        "Something about your body you appreciate",
        "A simple pleasure you had recently",
        "A personal quality you're glad to have",
        "Something that went better than expected",
    ], 3)
    st.markdown('<div style="color:#7a8aa0;font-size:0.84rem;margin-bottom:12px">'
                'Name <em>why</em> it matters — that\'s what shifts the brain. Vague gratitude doesn\'t work as well.</div>',
                unsafe_allow_html=True)
    with st.form("gratitude_form"):
        entries = [st.text_input(f"{i+1}. {p}", placeholder="Be specific…", key=f"grat_{i}")
                   for i,p in enumerate(prompts)]
        submitted = st.form_submit_button("💛 Save", use_container_width=True)
    if submitted:
        saved = [e for e in entries if e.strip()]
        if saved: db.log_gratitude(saved); st.success(f"Saved {len(saved)} entry/entries.")
        else: st.warning("Add at least one entry.")

    recent = db.get_gratitude(14)
    if recent:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        by_date = defaultdict(list)
        for g in recent: by_date[g["date"]].append(g["entry"])
        for d in sorted(by_date.keys(), reverse=True)[:4]:
            st.markdown(f'<div style="color:#7a8aa0;font-size:0.75rem;margin-top:10px">{d}</div>', unsafe_allow_html=True)
            for e in by_date[d]:
                st.markdown(f'<div class="card card-amber" style="padding:8px 14px;margin:2px 0;font-size:0.86rem">💛 {e}</div>', unsafe_allow_html=True)

def _reflect_patterns(db):
    try:
        import plotly.graph_objects as go
        PLOTLY = True
    except ImportError:
        PLOTLY = False

    days    = st.select_slider("Time window", [7,14,30], value=14, format_func=lambda x: f"{x} days")
    history = db.get_mood_history(days)
    if not history:
        st.markdown('<div class="card" style="text-align:center;padding:28px;color:#7a8aa0">No check-ins yet. Patterns appear after a few days.</div>', unsafe_allow_html=True)
        return

    dates   = [r["date"] for r in history]
    scores  = [r["score"] or 0 for r in history]
    emotions= [r["emotion"] or "neutral" for r in history]
    avg     = round(sum(scores)/len(scores),1) if scores else 0
    top_em  = Counter(emotions).most_common(1)[0][0].title() if emotions else "—"

    c1,c2,c3 = st.columns(3)
    with c1: _metric(f"{avg}/10","Avg mood")
    with c2: _metric(top_em,"Top emotion")
    with c3: _metric(str(len(history)),"Check-ins")

    if PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=scores, mode="lines+markers",
            line=dict(color=P()["color"], width=2), marker=dict(size=6, color=P()["color"]),
            fill="tozeroy", fillcolor=f"rgba(111,168,220,0.07)"))
        fig.update_layout(height=180, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7a8aa0", family="DM Sans"),
            margin=dict(l=0,r=0,t=8,b=0),
            yaxis=dict(range=[0,10], gridcolor="#252f45"), xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD
# ═══════════════════════════════════════════════════════════════════════════════

def page_build(db: Database):
    st.markdown("## Build habits")
    tab1, tab2 = st.tabs(["🎯 Goals", "📈 Progress"])
    with tab1: _build_goals(db)
    with tab2: _build_progress(db)

def _build_goals(db):
    cats = ["Mental health","Relationships","Work / Study","Physical health","Habits","Creativity","Other"]
    active    = db.get_goals(completed=False)
    completed = db.get_goals(completed=True)

    if not active:
        st.markdown('<div class="card" style="text-align:center;padding:22px;color:#7a8aa0">No active goals. Add one below.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:#7a8aa0;font-size:0.76rem;margin-bottom:6px">{len(active)} active · {len(completed)} done</div>', unsafe_allow_html=True)
        for g in active:
            c1,c2,c3 = st.columns([7,1,1])
            with c1:
                st.markdown(f'<div class="card card-green" style="padding:9px 14px;margin:3px 0"><div style="font-size:0.9rem">{g["title"]}</div><div style="font-size:0.72rem;color:#7a8aa0">{g["category"]}</div></div>', unsafe_allow_html=True)
            with c2:
                if st.button("✓", key=f"done_{g['id']}"): db.complete_goal(g["id"]); st.rerun()
            with c3:
                if st.button("✕", key=f"del_{g['id']}"): db.delete_goal(g["id"]); st.rerun()

    with st.form("goal_form"):
        title = st.text_input("New goal", placeholder="Specific: 'Walk 10 min every morning'")
        cat   = st.selectbox("Category", cats)
        if st.form_submit_button("Add", use_container_width=True):
            if title.strip(): db.add_goal(title.strip(), cat); st.rerun()
            else: st.warning("Enter a goal title.")

    if completed:
        with st.expander(f"✅ {len(completed)} completed"):
            for g in completed[-10:]:
                st.markdown(f'<div style="color:#52c788;font-size:0.85rem;padding:3px 0;opacity:0.7">✔ {g["title"]}</div>', unsafe_allow_html=True)

def _build_progress(db):
    try:
        import plotly.graph_objects as go
        PLOTLY = True
    except ImportError:
        PLOTLY = False

    streak  = db.get_streak()
    stats7  = db.get_mood_stats(7)
    stats30 = db.get_mood_stats(30)
    grats   = db.get_gratitude(30)
    done    = len(db.get_goals(completed=True))
    total_g = done + len(db.get_goals(completed=False))

    st.markdown(f"""<div class="card card-amber" style="text-align:center;padding:18px">
    <div style="font-family:Lora,serif;font-size:2.6rem;color:#d4a843;line-height:1">{streak['current']}</div>
    <div style="color:#7a8aa0;font-size:0.75rem;margin-top:3px">day streak 🔥 · best: {streak['longest']} · total: {streak['total']}</div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: _metric(f"{stats7['avg_score']}/10","Avg mood")
    with c2: _metric(str(len(grats)),"Gratitudes")
    with c3: _metric(str(done),"Goals done")
    with c4: _metric(f"{int(done/total_g*100) if total_g else 0}%","Completion")

    if stats30["count"] > 0:
        history = db.get_mood_history(30)
        if history and PLOTLY:
            ems = [r["emotion"] for r in history if r.get("emotion")]
            em_count = Counter(ems)
            fig = go.Figure(go.Bar(x=list(em_count.keys()), y=list(em_count.values()),
                marker_color=["#6fa8dc","#7ec8a0","#d4a843","#c97b84","#9b8ec4","#5ba8a0","#c4906e"][:len(em_count)]))
            fig.update_layout(height=190, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7a8aa0",family="DM Sans"), margin=dict(l=0,r=0,t=8,b=0),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#252f45"))
            st.plotly_chart(fig, use_container_width=True)

    # Recent assessments
    recent_a = db.get_assessments(limit=5)
    if recent_a:
        st.markdown("**Recent assessments**")
        for a in recent_a:
            st.markdown(f'<div style="font-size:0.8rem;color:#7a8aa0;padding:2px 0">'
                        f'<span style="color:#a0b0c0">{a["tool"]}</span> — '
                        f'{a["score"]:.0f} ({a["interpretation"]}) — {a["date"]}</div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CRISIS VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def page_crisis():
    st.markdown("## You are not alone.")
    st.markdown('<div style="color:#a0aec0;margin-bottom:16px">What you\'re feeling right now is not permanent. Trained people are available right now.</div>', unsafe_allow_html=True)
    render_crisis_lines()
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("**While you prepare to call, try this:**")
    _tool_box()

def render_crisis_lines():
    profile = P()
    lines   = profile["crisis_lines"]
    st.markdown('<div class="crisis-strip">', unsafe_allow_html=True)
    st.markdown('<div style="color:#e0778a;font-size:0.86rem;font-weight:500;margin-bottom:8px">You are not alone. Trained people are available 24/7.</div>', unsafe_allow_html=True)
    for name, number in lines:
        st.markdown(f'<div class="crisis-line"><span style="color:#c0cce0">{name}</span>'
                    f'<span class="crisis-number">{number}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

def page_settings():
    st.markdown("## Settings")
    st.markdown("#### Groq API Key")
    st.markdown('<div style="color:#7a8aa0;font-size:0.84rem;margin-bottom:8px">Powers the AI companion. Free at <a href="https://console.groq.com" target="_blank" style="color:#6fa8dc">console.groq.com</a></div>', unsafe_allow_html=True)

    if st.session_state.api_key:
        st.success("✅ Connected")
        if st.button("Remove key"): st.session_state.api_key = ""; st.rerun()
    else:
        key = st.text_input("Paste key", type="password", placeholder="gsk_...")
        if key: st.session_state.api_key = key; st.success("Saved."); st.rerun()

    st.markdown("---")
    st.markdown("#### Profile")
    st.markdown(f'Current: **{P()["icon"]} {P()["label"]}**')
    if st.button("Change profile"): st.session_state.view = "profile_select"; st.rerun()

    st.markdown("---")
    st.markdown("""<div class="card" style="font-size:0.82rem;color:#7a8aa0;line-height:1.7">
    All data is stored locally in <code>mindful_pro.db</code>. Nothing leaves your device
    except chat messages sent to Groq (if you use the AI companion).<br><br>
    Mindful Pro is a supportive tool — not a replacement for therapy or crisis services.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _metric(value, label):
    st.markdown(f'<div class="m-tile"><div class="m-val">{value}</div>'
                f'<div class="m-lbl">{label}</div></div>', unsafe_allow_html=True)

def _em_emoji(em):
    return {"anxious":"😰","sad":"😢","angry":"😠","overwhelmed":"😓","exhausted":"😴",
            "confused":"🤔","hopeful":"🙂","happy":"😊","neutral":"😐"}.get(em,"😐")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    inject_css(P()["color"])
    db  = get_db()
    view = st.session_state.view

    if view == "profile_select":
        page_profile_select()
        return

    render_topbar()

    if   view == "home":     page_home(db)
    elif view == "right_now":page_right_now()
    elif view == "talk":     page_talk(db)
    elif view == "reflect":  page_reflect(db)
    elif view == "build":    page_build(db)
    elif view == "assess":   page_assess(db)
    elif view == "learn":    page_learn()
    elif view == "safety":   page_safety(db)
    elif view == "crisis":   page_crisis()
    elif view == "settings": page_settings()
    else:
        st.session_state.view = "home"; st.rerun()

if __name__ == "__main__":
    main()