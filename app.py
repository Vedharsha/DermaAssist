import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from PIL import Image
import requests, io, numpy as np, json, os
from streamlit_cropper import st_cropper

OPENWEATHER_API_KEY = "api key"
MODEL_PATH          = "trained_model.pth"

EMBEDDING_DIM = 128
IMG_SIZE      = 240

DISEASE_CLASSES = [
    'Acne', 'Basal Cell Carcinoma', 'Benign tumors', 'Bullous', 'Candidiasis',
    'Dermatofibroma', 'DrugEruption', 'Eczema', 'Hailey-Hailey Disease', 'Herpes Simplex',
    'Impetigo', 'Infestations Bites', 'Keratosis', 'Larva Migrans', 'Leprosy', 'Lichen Disorder',
    'Lupus Erythematosus', 'Melanoma', 'Moles', 'Molluscum Contagiosum', 'Nevus', 'Normal',
    'Pityriasis Rosea', 'Psoriasis', 'Rosacea', 'Squamous cell carcinoma', 'Sun Sunlight Damage',
    'Tinea', 'Vascular Tumors or Lesion', 'Vasculitis', 'Vitiligo', 'Warts'
]

# ──────────────────────────────────────────────
# Disease-specific symptom weights for severity scoring
# Each symptom key maps to how much it contributes to severity (1-10 scale)
# ──────────────────────────────────────────────

DISEASE_SYMPTOM_WEIGHTS = {
    'Acne': {
        'comedones': 2, 'papules': 3, 'pustules': 4,
        'cystic_lesions': 9, 'oily_skin': 1, 'itching': 2,
        'hormonal_trigger': 2
    },
    'Melanoma': {
        'asymmetrical_mole': 8, 'irregular_border': 7, 'multiple_colors': 7,
        'diameter_larger_than_pencil': 6, 'itching': 5, 'changing': 10,
        'family_history_melanoma': 5
    },
    'Psoriasis': {
        'silvery_scales': 3, 'plaques': 4, 'itching': 3,
        'bleeding_when_scraped': 6, 'nails_affected': 5,
        'joint_pain': 7
    },
    'Eczema': {
        'intense_itching': 5, 'dry_skin': 2, 'redness': 3,
        'small_raised_bumps': 3, 'cracking_bleeding': 7,
        'swelling': 5
    },
    'Herpes Simplex': {
        'blisters': 5, 'tingling_burning': 3, 'painful_ulcers': 7,
        'crusting': 3, 'fever': 6, 'swollen_lymph_nodes': 5
    },
    'Basal Cell Carcinoma': {
        'pearl_nodule': 5, 'bleeding_ulcer': 8, 'shiny_bump': 4,
        'central_depression': 6, 'telangiectasia': 3,
        'ulceration': 9, 'sun_exposure': 3
    },
    'Squamous cell carcinoma': {
        'red_scaly_patch': 4, 'raised_nodule': 5, 'bleeding_ulcer': 9,
        'open_sore': 8, 'rapid_growth': 9, 'itching_pain': 4
    },
    'Tinea': {
        'circular_patches': 3, 'red_borders': 3, 'scaling': 3,
        'itching': 3, 'burning': 3, 'multiple_patches': 5
    },
    'Warts': {
        'small_bumps': 2, 'rough_texture': 2, 'black_dots': 3,
        'clusters': 4, 'recurring': 6
    },
    'Vitiligo': {
        'depigmented_patches': 4, 'sun_sensitivity': 3,
        'hair_whitening': 5, 'itching': 2
    },
    'Impetigo': {
        'honey_crusted': 3, 'blisters': 5, 'pustules': 5,
        'red_base': 3, 'rapid_spread': 9, 'itching': 2,
        'fever': 7
    },
    'Rosacea': {
        'persistent_redness': 3, 'flushing': 3, 'visible_blood_vessels': 4,
        'bumps_pustules': 5, 'facial_burning': 4, 'thickened_skin': 7,
        'eye_symptoms': 6
    },
    'Candidiasis': {
        'white_patches': 4, 'red_irritation': 3, 'itching_burning': 4,
        'discharge': 4, 'maceration': 5, 'painful': 5
    },
    'Dermatofibroma': {
        'firm_nodule': 3, 'itching': 2, 'stable': -2
    },
    'Moles': {
        'uniform_color': -2, 'stable': -3, 'regular_border': -2
    },
    'Keratosis': {
        'rough_texture': 3, 'itching': 2, 'multiple_lesions': 4,
        'location_sun_exposed': 3
    },
    'Lupus Erythematosus': {
        'butterfly_rash': 6, 'photosensitivity': 4, 'discoid_lesions': 5,
        'oral_ulcers': 6, 'joint_pain': 5, 'fatigue': 4,
        'fever': 7, 'diagnosed_lupus': 5
    },
    'Vasculitis': {
        'palpable_purpura': 7, 'lower_extremity': 4, 'itching_burning': 3,
        'joint_pain': 5, 'systemic_symptoms': 8, 'worsening_pattern': 6
    },
    'DrugEruption': {
        'rash_onset': 4, 'widespread_distribution': 6, 'itching': 3,
        'fever': 7, 'systemic_symptoms': 8
    },
    'Bullous': {
        'fluid_filled_blisters': 6, 'erosions': 7, 'itching_burning': 4,
        'mucosal_involvement': 8, 'widespread': 8
    },
    'Hailey-Hailey Disease': {
        'recurrent_blisters': 6, 'hyperhidrosis': 3, 'itching_burning': 4,
        'secondary_infection': 8
    },
    'Infestations Bites': {
        'multiple_bites': 3, 'itching_intense': 5, 'linear_pattern': 5,
        'burrows': 7, 'pustules_crusts': 5
    },
    'Larva Migrans': {
        'linear_track': 6, 'intense_itching': 5, 'raised_tunnel': 6,
        'blisters_pustules': 5, 'erythematous': 3
    },
    'Leprosy': {
        'hypopigmented_patches': 4, 'numbness': 8, 'nerve_involvement': 9,
        'nodules': 5, 'eye_involvement': 8
    },
    'Lichen Disorder': {
        'papules': 4, 'itching_burning': 5, 'white_patches': 4,
        'oral_involvement': 6, 'genital_involvement': 7
    },
    'Molluscum Contagiosum': {
        'pearly_lesions': 3, 'clusters': 4, 'itching': 2,
        'number_lesions': 3
    },
    'Nevus': {
        'stable': -3, 'flat_raised': 1, 'cosmetic_concern': 1
    },
    'Pityriasis Rosea': {
        'herald_patch': 3, 'secondary_eruption': 4, 'scaly_patches': 3,
        'itching': 3, 'spreading': 5
    },
    'Sun Sunlight Damage': {
        'photodamage_signs': 4, 'solar_lentigines': 3, 'rough_texture': 3,
        'telangiectasia': 3, 'sunburn_history': 4
    },
    'Vascular Tumors or Lesion': {
        'red_purple_color': 4, 'raised_nodule': 4, 'bleeding': 7,
        'growing': 6
    },
    'Benign tumors': {
        'raised_growth': 2, 'stable': -2, 'cosmetic_concern': 1
    },
}

# ──────────────────────────────────────────────
# User-friendly text simplifier
# ──────────────────────────────────────────────

def simplify_recommendation(text):
    """Convert medical jargon to plain, friendly language."""
    replacements = {
        # Medication names → plain
        "benzoyl peroxide 2.5%": "a gentle acne wash (benzoyl peroxide 2.5%)",
        "benzoyl peroxide 5-10%": "a stronger acne wash (benzoyl peroxide 5-10%)",
        "benzoyl peroxide 10%": "a strong acne wash",
        "benzoyl peroxide": "acne wash",
        "topical terbinafine 1% cream": "an antifungal cream (terbinafine) from the pharmacy",
        "oral terbinafine 250mg": "antifungal tablets (terbinafine) — get these from a doctor",
        "mupirocin 2% ointment": "an antibiotic ointment (mupirocin) from a pharmacy",
        "oral flucloxacillin 500mg": "antibiotic capsules (flucloxacillin) — prescribed by your doctor",
        "iv flucloxacillin": "antibiotic drip in hospital",
        "isotretinoin": "a strong acne tablet (isotretinoin) — needs a doctor's prescription",
        "hydroxychloroquine": "a lupus medication (hydroxychloroquine)",
        "topical corticosteroid": "a steroid cream from your doctor",
        "mometasone furoate 0.1%": "a steroid cream (mometasone)",
        "clobetasol propionate 0.05%": "a strong steroid cream",
        "clobetasol 0.05%": "a strong steroid cream",
        "tretinoin": "a vitamin A cream (tretinoin)",
        "adapalene": "a retinol gel (adapalene) — available at pharmacies",
        "acyclovir": "an antiviral tablet (acyclovir)",
        "valacyclovir": "an antiviral tablet (valacyclovir)",
        "methotrexate": "an immune-calming tablet (methotrexate) — needs specialist prescription",
        "spironolactone": "a hormone-balancing tablet — needs a doctor",
        "doxycycline 100mg": "an antibiotic tablet (doxycycline) — get this from a doctor",
        "minocycline 100mg": "an antibiotic tablet (minocycline) — get this from a doctor",
        "oral prednisolone": "steroid tablets from your doctor",
        "prednisolone": "steroid tablets",
        "chlorhexidine 0.05%": "antiseptic wash (chlorhexidine) from a pharmacy",
        "salicylic acid 2%": "salicylic acid (found in many pharmacy face washes)",
        "azelaic acid": "azelaic acid cream (available at pharmacies)",
        "calamine lotion": "calamine lotion from the pharmacy",
        "griseofulvin": "antifungal tablets (griseofulvin) — from a doctor",
        "fluconazole": "antifungal tablets (fluconazole)",
        "itraconazole": "antifungal tablets (itraconazole) — from a doctor",
        "permethrin 5% cream": "a scabies treatment cream (permethrin) from a pharmacy",
        "ivermectin": "an antiparasite tablet (ivermectin) — from a doctor",
        "albendazole": "an antiparasite tablet (albendazole) — from a doctor",
        "cemiplimab": "a cancer treatment drug given by specialists",
        "pembrolizumab": "a cancer treatment drug given by specialists",
        "cyclophosphamide": "a strong immune treatment given in hospital",
        "ruxolitinib": "a newer skin treatment cream — ask your dermatologist",
        "tacrolimus 0.1%": "an immune-calming cream (tacrolimus) — prescription needed",
        "pimecrolimus": "an immune-calming cream — prescription needed",
        "imiquimod 5% cream": "an immune-boosting cream (imiquimod) — prescription needed",
        "topical retinoid": "a vitamin A cream (available on prescription)",
        "emollient": "a good moisturiser",
        "emollients": "moisturisers",
        "occlusive emollient": "a thick, protective moisturiser",
        "ceramide": "ceramide moisturiser (e.g. CeraVe)",
        "colloidal oatmeal": "oatmeal-based lotion (e.g. Aveeno)",
        "petroleum-based": "petroleum jelly (Vaseline) based",

        # Clinical procedures → plain
        "excisional biopsy": "surgical removal of the lesion for testing",
        "mohs micrographic surgery": "a specialist surgery to remove skin cancer precisely",
        "mohs surgery": "specialist cancer removal surgery",
        "cryotherapy": "freezing the lesion (done by a doctor)",
        "phototherapy": "light therapy at a clinic",
        "nb-uvb": "a special UV light therapy at a clinic",
        "psoralen-uva": "light therapy combined with a tablet",
        "curettage and cauterization": "scraping and burning the lesion — done by a doctor",
        "electrodesiccation": "an electrical treatment done by a doctor",
        "laser therapy": "laser treatment at a clinic",
        "patch testing": "allergy testing done at a clinic",
        "skin biopsy": "a small skin sample taken by a doctor for testing",
        "mdt": "a specialist team of doctors",
        "lft": "liver function blood test",
        "lfts": "liver function blood tests",
        "tsh": "thyroid blood test",
        "tpo antibodies": "thyroid antibody blood test",
        "fbc": "a full blood count test",
        "crp": "an inflammation blood test",
        "mc&s": "a swab test to check for infection",
        "pcr": "a lab test",
        "spf 50+": "SPF 50+ sunscreen",
        "spf 30+": "SPF 30+ sunscreen",
        "spf 50": "SPF 50 sunscreen",

        # Medical terms → plain
        "pilosebaceous": "hair follicle and oil gland",
        "comedones": "blackheads and whiteheads",
        "comedogenic": "pore-blocking",
        "non-comedogenic": "non-pore-blocking",
        "keratolytic": "helps remove dead skin",
        "bacteriostatic": "stops bacteria from growing",
        "pruritic": "very itchy",
        "erythematous": "red and inflamed",
        "lesion": "the affected skin area",
        "lesions": "the affected skin areas",
        "pustules": "pus-filled spots",
        "papules": "small red bumps",
        "nodules": "larger bumps under the skin",
        "bullae": "large fluid-filled blisters",
        "vesicles": "small fluid-filled blisters",
        "plaques": "raised, thickened patches of skin",
        "erythema": "skin redness",
        "excoriation": "broken skin from scratching",
        "hyperpigmentation": "dark marks left after healing",
        "post-inflammatory hyperpigmentation": "dark marks left after a breakout heals",
        "depigmentation": "loss of skin colour",
        "onychomycosis": "a fungal nail infection",
        "tinea capitis": "scalp ringworm",
        "tinea corporis": "ringworm on the body",
        "tinea pedis": "athlete's foot",
        "ecthyma": "a deeper form of impetigo",
        "ecthymatous": "deep and crusted",
        "bacteraemia": "bacteria in the bloodstream — serious",
        "lymphangitis": "infection spreading along the lymph channels — see a doctor urgently",
        "lymphadenopathy": "swollen lymph nodes/glands",
        "perineural invasion": "when cancer spreads along nerves",
        "metastatic": "cancer that has spread to other parts of the body",
        "immunocompromised": "having a weakened immune system",
        "immunosuppressive": "a medicine that lowers the immune system",
        "hepatotoxic": "can affect the liver",
        "nephrotoxic": "can affect the kidneys",
        "teratogenic": "can harm an unborn baby",
        "contraindication": "a reason not to use a treatment",
        "contraindicated": "should not be used",
        "tachyphylaxis": "when your skin gets used to a cream and it stops working as well",
        "prophylaxis": "preventive treatment",
        "sensitisation": "developing an allergy over time",
        "allergen": "something that triggers an allergy",
        "antimicrobial": "germ-killing",
        "antifungal": "fungus-fighting",
        "antibacterial": "bacteria-fighting",
        "antiviral": "virus-fighting",
        "topical": "applied directly to the skin",
        "systemic": "taken by mouth or injection to work throughout the body",
        "adjuvant": "supporting",
        "bsa": "body surface area",
        "uv": "sunlight (UV rays)",
        "uvb": "a type of sunlight (UVB rays)",
        "uvr": "sunlight rays",
        "photosensitivity": "sensitivity to sunlight",
        "photoprotection": "protecting your skin from the sun",
        "broad-spectrum": "protecting against all types of UV rays",
        "mineral sunscreen": "a physical sunscreen with zinc oxide",
        "q2h": "every 2 hours",
        "q4-6h": "every 4–6 hours",
        "q60-90 minutes": "every 60–90 minutes",
        "bd": "twice a day",
        "tds": "three times a day",
        "tid": "three times a day",
        "qid": "four times a day",
        "once daily": "once a day",
        "twice daily": "twice a day",
        "x 1-2 weeks": "for 1–2 weeks",
        "x 2 weeks": "for 2 weeks",
        "x 4 weeks": "for 4 weeks",
        "x 6 weeks": "for 6 weeks",
        "x 7-10 days": "for 7–10 days",
        "clinical assessment": "a doctor's check-up",
        "dermatology referral": "a referral to a skin specialist",
        "dermatology consultation": "a visit to a skin specialist",
        "specialist review": "a specialist doctor check-up",
        "gp review": "a visit to your family doctor",
        "consult a dermatologist": "see a skin specialist",
        "consult your physician": "speak to your doctor",
        "over-the-counter": "available at a pharmacy without prescription",
        "iplEdge program": "a special safety programme for this medication",
        "cognitive behavioral": "talking therapy",
        "psychotherapy": "talking therapy with a professional",
        "seborrheic": "related to oily skin areas",
        "stratum corneum": "the outermost layer of skin",
        "skin barrier": "the protective outer layer of your skin",
        "skin barrier function": "how well your skin protects itself",
        "transepidermal water loss": "moisture escaping through the skin",
        "ph 4.5-5.5": "slightly acidic (gentle on skin)",
    }

    result = text
    for medical, plain in replacements.items():
        result = result.replace(medical, plain)
        result = result.replace(medical.title(), plain)
        result = result.replace(medical.upper(), plain)
    return result


def make_friendly_recommendations(recs):
    """
    Take the raw recommendation dict and return a new dict
    with simplified, warm, plain-English text.
    """
    friendly = {}
    for key, value in recs.items():
        if key == "personalization":
            friendly[key] = value
            continue
        if isinstance(value, list):
            friendly[key] = [simplify_recommendation(item) for item in value]
        else:
            friendly[key] = value
    return friendly


# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────

@st.cache_resource
def load_configs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs = {}
    for file in ["disease_symptoms.json", "disease_recommendations.json"]:
        path = os.path.join(base_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                configs[file] = json.load(f)
        except Exception as e:
            st.warning(f"Could not load {file}: {e}")
            configs[file] = {}
    return (
        configs.get("disease_symptoms.json", {}),
        configs.get("disease_recommendations.json", {})
    )

DISEASE_SYMPTOMS_CONFIG, DISEASE_RECOMMENDATIONS_CONFIG = load_configs()

# ──────────────────────────────────────────────
# Model — mirrors notebook's DermaNet exactly
# ──────────────────────────────────────────────

class DermaNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128,
                 dropout_rate: float = 0.4):
        super().__init__()
        backbone  = efficientnet_b1(weights=None)
        in_feats  = backbone.classifier[1].in_features

        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        self.embedding_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_feats, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x, return_embedding: bool = False):
        x      = self.features(x)
        x      = self.avgpool(x)
        x      = x.flatten(1)
        emb    = self.embedding_head(x)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


@st.cache_resource
def load_model(path: str):
    model = DermaNet(len(DISEASE_CLASSES), embedding_dim=EMBEDDING_DIM, dropout_rate=0.4)
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.1)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


OOD_CONF_THRESHOLD = 0.30
OOD_VAR_THRESHOLD  = 0.05
MC_DROPOUT_PASSES  = 20


def _enable_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _mc_dropout_variance(model: DermaNet, tensor: torch.Tensor,
                          n_passes: int = MC_DROPOUT_PASSES) -> float:
    model.eval()
    _enable_dropout(model)
    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    model.eval()
    all_probs = np.array(all_probs)
    variance  = all_probs.var(axis=0).mean(axis=1)
    return float(variance[0])


def predict_disease(model: DermaNet, image: Image.Image):
    tensor = preprocess_image(image)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze()

    confidence, class_idx = torch.max(probs, dim=0)
    conf_val  = confidence.item()
    probs_np  = probs.numpy()

    mc_variance = _mc_dropout_variance(model, tensor, MC_DROPOUT_PASSES)

    low_confidence = conf_val  < OOD_CONF_THRESHOLD
    high_variance  = mc_variance > OOD_VAR_THRESHOLD
    ood_flag       = low_confidence or high_variance

    return (
        DISEASE_CLASSES[class_idx.item()],
        conf_val,
        probs_np,
        ood_flag,
        mc_variance,
        low_confidence,
        high_variance,
    )

# ──────────────────────────────────────────────
# Weather + AQI
# ──────────────────────────────────────────────

def get_weather_with_aqi(city: str, api_key: str) -> dict:
    mock = {
        "temp": 28, "humidity": 65,
        "aqi": "moderate", "aqi_value": 75,
        "description": "simulated (no API key)",
        "city": city or "Unknown",
        "source": "mock"
    }
    if not api_key:
        return mock

    try:
        w_url  = (f"https://api.openweathermap.org/data/2.5/weather"
                  f"?q={city}&appid={api_key}&units=metric")
        w_resp = requests.get(w_url, timeout=8)
        if w_resp.status_code != 200:
            return {**mock, "description": f"city not found ({w_resp.status_code})"}

        w_data = w_resp.json()
        lat    = w_data["coord"]["lat"]
        lon    = w_data["coord"]["lon"]
        weather_data = {
            "temp":        round(w_data["main"]["temp"], 1),
            "humidity":    w_data["main"]["humidity"],
            "description": w_data["weather"][0]["description"],
            "city":        w_data["name"],
            "source":      "live",
        }
    except Exception as e:
        return {**mock, "description": f"weather API error: {e}"}

    try:
        aqi_url  = (f"https://api.openweathermap.org/data/2.5/air_pollution"
                    f"?lat={lat}&lon={lon}&appid={api_key}")
        aqi_resp = requests.get(aqi_url, timeout=8)
        if aqi_resp.status_code == 200:
            aqi_raw = aqi_resp.json()["list"][0]["main"]["aqi"]
            aqi_map = {1: "good", 2: "fair", 3: "moderate", 4: "poor", 5: "very poor"}
            weather_data["aqi"]       = aqi_map.get(aqi_raw, "moderate")
            weather_data["aqi_value"] = aqi_raw * 50
        else:
            weather_data["aqi"]       = "moderate"
            weather_data["aqi_value"] = 75
    except Exception:
        weather_data["aqi"]       = "moderate"
        weather_data["aqi_value"] = 75

    return weather_data

# ──────────────────────────────────────────────
# Severity — FIXED: uses disease-specific symptom weights
# ──────────────────────────────────────────────

def compute_severity_score(disease, symptoms, personal, weather):
    if disease == "Normal":
        return 0

    # Base score by disease risk level
    score = (
        30 if disease in ['Melanoma', 'Basal Cell Carcinoma', 'Squamous cell carcinoma',
                          'Lupus Erythematosus', 'Vasculitis', 'Leprosy']
        else 20 if disease in ['Bullous', 'Hailey-Hailey Disease', 'DrugEruption',
                               'Larva Migrans', 'Herpes Simplex']
        else 15 if disease in ['Psoriasis', 'Eczema', 'Rosacea', 'Lichen Disorder',
                               'Candidiasis', 'Impetigo', 'Infestations Bites',
                               'Vitiligo', 'Keratosis', 'Sun Sunlight Damage']
        else 5
    )

    # Use disease-specific weights if available
    disease_weights = DISEASE_SYMPTOM_WEIGHTS.get(disease, {})
    if disease_weights:
        for sym, weight in disease_weights.items():
            val = symptoms.get(sym, False)
            # Boolean checkboxes
            if isinstance(val, bool) and val:
                score += weight
            # Multi-checkbox (list) — score based on how many selected
            elif isinstance(val, list) and len(val) > 0:
                score += min(weight, len(val) * 2)
            # Slider values — scale by how high the value is relative to midpoint
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                pass  # sliders don't add directly to severity
    else:
        # Fallback for diseases with no specific weights
        for sym in ['itching', 'redness', 'swelling', 'bleeding', 'fever',
                    'pain', 'blisters', 'discharge']:
            score += 2 if symptoms.get(sym, False) else 0

    # Personal risk factors
    age = personal.get('age', 30)
    score += (8 if age > 60 or age < 5 else 4 if age > 50 else 0)
    score += 5  if personal.get('existing_conditions', '').strip() else 0
    score += 10 if personal.get('immunocompromised') else 0

    # Environmental factors
    temp, humidity = weather.get('temp', 28), weather.get('humidity', 65)
    score += 10 if (temp > 35 or humidity > 80) else 5 if (temp > 30 or humidity > 70) else 0

    aqi_value = weather.get('aqi_value', 75)
    score += 8 if aqi_value > 200 else 5 if aqi_value > 150 else 0

    return min(score, 100)


def score_to_severity(score):
    return "Mild" if score < 35 else "Moderate" if score < 65 else "Severe"

# ──────────────────────────────────────────────
# Recommendations — with user-friendly language
# ──────────────────────────────────────────────

def get_professional_recommendations(disease, severity, weather, skin_type, allergies,
                                     existing_conditions='', immune_detail=''):
    disease_recs  = DISEASE_RECOMMENDATIONS_CONFIG.get(disease, {})
    severity_recs = disease_recs.get(severity, {})

    if severity_recs:
        base_recs = {k: list(v) for k, v in severity_recs.items() if isinstance(v, list)}
        for key in ["immediate_care", "medications", "environmental_precautions",
                    "lifestyle_adjustments", "when_to_seek_medical_attention", "warnings"]:
            base_recs.setdefault(key, [])
    else:
        base_recs = {
            "immediate_care":                 ["Please visit a skin doctor (dermatologist) for a proper check-up."],
            "medications":                    ["Your doctor will recommend the right treatment after examining your skin."],
            "environmental_precautions":      ["Use sunscreen daily and keep the area clean and dry."],
            "lifestyle_adjustments":          ["Drink enough water, sleep well, and try to keep stress low."],
            "when_to_seek_medical_attention": ["See a doctor if your symptoms get worse or do not improve."],
            "warnings":                       []
        }

    temp = weather.get('temp', 28)
    if temp > 35:
        base_recs["environmental_precautions"].append(
            f"🌡️ It's very hot outside ({temp}°C) — stay in cool, air-conditioned spaces as much as possible. "
            "Heat can make your skin condition worse.")
        if disease in ["Acne", "Rosacea"]:
            base_recs["medications"].append(
                "In hot weather, switch to lighter gel-based creams instead of heavy ointments — they feel better and work just as well.")
    elif temp < 10:
        base_recs["environmental_precautions"].append(
            f"🧣 It's cold outside ({temp}°C) — use a thicker moisturiser more often and avoid long, hot showers as they dry out your skin.")

    humidity = weather.get('humidity', 65)
    if humidity > 80:
        base_recs["environmental_precautions"].append(
            f"💧 The air is very humid ({humidity}%) — use lighter, water-based creams and keep skin dry, "
            "especially in skin folds, to prevent fungal infections.")
    elif humidity < 30:
        base_recs["environmental_precautions"].append(
            f"🏜️ The air is very dry ({humidity}%) — moisturise more often and consider using a room humidifier to keep your skin from drying out.")

    aqi_value = weather.get('aqi_value', 75)
    if aqi_value > 200:
        base_recs["warnings"].append(
            f"🚨 The air quality outside is very poor (AQI {aqi_value}) — stay indoors as much as possible. "
            "Air pollution can make inflamed skin much worse.")
    elif aqi_value > 150:
        base_recs["environmental_precautions"].append(
            f"😷 Air quality is poor today (AQI {aqi_value}) — wear a mask outdoors and wash your face when you come back inside.")

    if skin_type == "Oily":
        base_recs["medications"].append(
            "Since your skin is oily, use light, water-based or gel products. Look for 'non-comedogenic' (non-pore-blocking) on the label.")
    elif skin_type == "Dry":
        base_recs["medications"].append(
            "Since your skin is dry, use rich, fragrance-free moisturisers with ceramides (e.g. CeraVe, Cetaphil) to help your skin stay hydrated.")
    elif skin_type == "Sensitive":
        base_recs["medications"].append(
            "Since your skin is sensitive, choose gentle, fragrance-free products with as few ingredients as possible. Always patch test on a small area first.")

    if allergies:
        allergies_str = ", ".join(allergies)
        base_recs["warnings"].append(
            f"🚫 You've told us you're allergic to: {allergies_str}. "
            "Always check the label of any cream, tablet, or lotion before using it.")
        base_recs["environmental_precautions"].append(
            f"Make sure any product prescribed to you does not contain {allergies_str}. Tell your pharmacist and doctor about these allergies.")
        if "Benzoyl peroxide" in allergies:
            base_recs["medications"] = [
                m for m in base_recs["medications"] if "benzoyl" not in m.lower()
            ]
            base_recs["medications"].append(
                "Since you're allergic to benzoyl peroxide, safe alternatives include salicylic acid 2–3%, azelaic acid, or dapsone gel — ask your pharmacist.")

    cond_text = (existing_conditions + " " + immune_detail).lower()
    CONDITION_RULES = {
        ("diabetes", "diabetic"): (
            "warnings",
            "🩸 You have diabetes — keep your blood sugar in check, as high sugar slows down skin healing "
            "and makes infections harder to treat. Let your doctor know before starting any new skin treatment."
        ),
        ("hypertension", "blood pressure", "bp"): (
            "warnings",
            "💊 You have high blood pressure — some medications (like ibuprofen or strong steroid tablets) "
            "can raise blood pressure further. Always mention this to your doctor."
        ),
        ("kidney", "renal", "ckd", "dialysis"): (
            "warnings",
            "🫘 You have a kidney condition — some skin medicines need lower doses when kidneys are not fully working. "
            "Do not start any new tablets without checking with your kidney doctor first."
        ),
        ("liver", "hepatitis", "cirrhosis", "jaundice"): (
            "warnings",
            "🫀 You have a liver condition — some antifungal and acne tablets can affect the liver. "
            "Stick to creams applied to the skin where possible, and check with your doctor before taking anything by mouth."
        ),
        ("heart", "cardiac", "heart failure", "arrhythmia"): (
            "warnings",
            "❤️ You have a heart condition — some antihistamines and antifungal tablets can affect your heart rhythm. "
            "Always tell your cardiologist about any new skin treatment."
        ),
        ("thyroid", "hypothyroid", "hyperthyroid"): (
            "lifestyle_adjustments",
            "🦋 You have a thyroid condition — thyroid problems can affect your skin's moisture and healing. "
            "Make sure your thyroid levels are well-controlled, as this directly helps your skin too."
        ),
        ("pregnant", "pregnancy", "breastfeeding", "lactating"): (
            "warnings",
            "🤰 You are pregnant or breastfeeding — many skin treatments including vitamin A creams, "
            "certain antibiotics, and strong steroids are not safe to use. Always check with your obstetrician before applying anything."
        ),
        ("asthma", "copd", "respiratory", "lung"): (
            "environmental_precautions",
            "🫁 You have a breathing condition — poor air quality and strong-smelling products (like sprays and perfumes) "
            "can affect both your lungs and skin. Wear a mask outdoors and avoid aerosol skincare products."
        ),
        ("autoimmune", "lupus", "rheumatoid", "crohn", "colitis"): (
            "warnings",
            "🛡️ You have an autoimmune condition — your immune system is already being managed carefully. "
            "Do not add any new immune-lowering skin treatments without your specialist's approval."
        ),
        ("cancer", "chemotherapy", "radiation", "oncology"): (
            "warnings",
            "🎗️ You are undergoing cancer treatment — chemotherapy makes your skin much more sensitive and prone to infection. "
            "Use only gentle, fragrance-free products and get approval from your oncologist before starting any skin treatment."
        ),
        ("hiv", "aids"): (
            "warnings",
            "🔴 You have HIV/AIDS — skin conditions can be more severe and unusual with a weakened immune system. "
            "Work closely with both your HIV specialist and a skin doctor for the best results."
        ),
        ("transplant", "immunosuppressant", "tacrolimus", "cyclosporine"): (
            "warnings",
            "💉 You've had an organ transplant or are on immune-lowering medicines — this puts you at much higher risk of skin cancers and infections. "
            "Get a full skin check every 3–6 months and never skip sun protection."
        ),
        ("steroid", "prednisone", "prednisolone", "corticosteroid"): (
            "warnings",
            "💊 You are on long-term steroid tablets — these can thin the skin and slow healing. "
            "Avoid adding strong steroid creams on top without a doctor's advice."
        ),
    }
    for keywords, (section, message) in CONDITION_RULES.items():
        if any(kw in cond_text for kw in keywords):
            base_recs[section].append(message)

    conditions_summary = existing_conditions.strip() if existing_conditions.strip() else "None reported"

    # Apply user-friendly simplification to all text
    friendly_recs = make_friendly_recommendations(base_recs)

    return {
        **friendly_recs,
        "personalization": {
            "disease":          disease,
            "severity":         severity,
            "temperature":      f"{weather.get('temp')}°C",
            "humidity":         f"{weather.get('humidity')}%",
            "aqi":              weather.get('aqi', 'moderate'),
            "skin_type":        skin_type,
            "allergies":        allergies if allergies else ["None reported"],
            "other_conditions": conditions_summary,
        }
    }

# ──────────────────────────────────────────────
# Symptom form
# ──────────────────────────────────────────────

def render_disease_specific_form(disease, form_container):
    if disease not in DISEASE_SYMPTOMS_CONFIG:
        st.warning(f"No symptom configuration found for '{disease}'.")
        return {}, {}, {}

    disease_config = DISEASE_SYMPTOMS_CONFIG[disease]
    ask_symptoms   = disease_config.get("ask_symptoms", True)
    ask_skin_type  = disease_config.get("ask_skin_type", True)
    ask_allergy    = disease_config.get("ask_allergy_history", False)

    symptoms_data, personal_data = {}, {}

    with form_container:
        if ask_symptoms and disease != "Normal":
            st.subheader("✓ Disease-Specific Clinical Symptoms")
            symptoms_list = disease_config.get("symptoms", [])

            checkboxes   = [s for s in symptoms_list if s["type"] == "checkbox"]
            sliders      = [s for s in symptoms_list if s["type"] == "slider"]
            radios       = [s for s in symptoms_list if s["type"] == "radio"]
            texts        = [s for s in symptoms_list if s["type"] == "text"]
            multi_checks = [s for s in symptoms_list if s["type"] == "multi_checkbox"]

            if checkboxes:
                cols = st.columns(2)
                for idx, s in enumerate(checkboxes):
                    symptoms_data[s["name"]] = cols[idx % 2].checkbox(
                        s["label"], key=f"{disease}_{s['name']}"
                    )
            for s in sliders:
                unit_label = f" ({s['unit']})" if s.get("unit") else ""
                symptoms_data[s["name"]] = st.slider(
                    s["label"] + unit_label,
                    min_value=s.get("min", 1), max_value=s.get("max", 10),
                    value=s.get("min", 1), key=f"{disease}_{s['name']}"
                )
            for s in radios:
                symptoms_data[s["name"]] = st.radio(
                    s["label"], options=s.get("options", []),
                    key=f"{disease}_{s['name']}", horizontal=True
                )
            for s in texts:
                symptoms_data[s["name"]] = st.text_input(
                    s["label"], placeholder=s.get("placeholder", ""),
                    key=f"{disease}_{s['name']}"
                )
            for s in multi_checks:
                st.markdown(f"**{s['label']}**")
                selected = []
                cols2 = st.columns(3)
                for i, opt in enumerate(s.get("options", [])):
                    if cols2[i % 3].checkbox(opt, key=f"{disease}_{s['name']}_{opt}"):
                        selected.append(opt)
                symptoms_data[s["name"]] = selected

        st.divider()
        st.subheader("👤 Patient Demographics")
        col1, col2 = st.columns(2)

        with col1:
            personal_data['age']  = st.number_input("Age", 0, 120, 30, key=f"{disease}_age")
            personal_data['city'] = st.text_input(
                "📍 Your City (for live weather & AQI)",
                placeholder="e.g. Chennai",
                help="Enter your city to fetch real-time temperature, humidity and air quality.",
                key=f"{disease}_city"
            )
        with col2:
            if ask_skin_type:
                personal_data['skin_type'] = st.selectbox(
                    "Skin Type",
                    ["Select", "Dry", "Oily", "Sensitive", "Combination", "Normal"],
                    key=f"{disease}_skin_type"
                )
            else:
                personal_data['skin_type'] = "Not applicable"

            immune_answer = st.selectbox(
                "Do you have any condition that weakens your immune system?",
                [
                    "No / Not sure",
                    "Yes — I have diabetes",
                    "Yes — I am on steroids or chemotherapy",
                    "Yes — I have HIV/AIDS",
                    "Yes — I had an organ transplant",
                    "Yes — other (e.g. cancer, autoimmune disease)",
                ],
                help="Conditions like diabetes, HIV, cancer treatment, or long-term steroid use can make infections harder to fight off.",
                key=f"{disease}_immuno"
            )
            personal_data['immunocompromised'] = immune_answer != "No / Not sure"
            personal_data['immune_detail']     = immune_answer

        if ask_allergy:
            personal_data['allergies'] = st.multiselect(
                "Known drug/ingredient allergies",
                ["Benzoyl peroxide", "Salicylic acid", "Retinoids", "Fragrance", "Sulfates", "Other"],
                key=f"{disease}_allergies_list"
            )
        else:
            personal_data['allergies'] = []

        personal_data['existing_conditions'] = st.text_input(
            "Any other health conditions? (optional)",
            placeholder="e.g. thyroid disorder, kidney disease, heart condition",
            help="Skip this if you already answered the immune system question above.",
            key=f"{disease}_conditions"
        )

    return symptoms_data, personal_data, {}

# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Dermatological Diagnosis System v4.3",
        page_icon="🩺",
        layout="wide"
    )
    st.title("🩺 Dermatological Disease Detection and Environment-Based Skin Health Assistance Using Deep Learning")
    st.caption(
        "AI-assisted diagnosis • Severity stratification • Live weather & AQI integration "
        "• Skin-type aware • Allergy-safe recommendations"
    )
    st.divider()

    with st.sidebar:
        st.header("⚙️ System Configuration")
        model_path = st.text_input("Model checkpoint path", value=MODEL_PATH)
        api_key    = st.text_input("OpenWeatherMap API Key", value=OPENWEATHER_API_KEY, type="password")
        st.caption("API key is used to fetch real-time temperature, humidity, and AQI for your location.")

    # ── Step 1: Image upload ──
    st.header("📷 Step 1 — Upload Skin Image")

    uploaded = st.file_uploader(
        "Upload a high-resolution dermatological image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is None:
        st.info("Please upload a skin lesion image to begin AI analysis.")
        st.stop()

    file_id = f"{uploaded.name}_{uploaded.size}"
    original_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    img_col, opt_col = st.columns([1, 1])
    with img_col:
        st.image(original_image, caption="Uploaded Image", use_container_width=True)
    with opt_col:
        st.markdown("""
        <div style="
            background: rgba(99,102,241,0.07);
            border: 1px solid rgba(99,102,241,0.2);
            border-left: 3px solid #6366f1;
            border-radius: 0 12px 12px 0;
            padding: 1rem 1.1rem;
            font-size: 0.84rem;
            line-height: 1.75;
            margin-bottom: 1rem;
        ">
        <strong style="font-size:0.92rem;">✂️ Optional: Crop to Disease Area</strong><br><br>
        Tight cropping removes background noise and can improve AI accuracy.<br><br>
        • Include <strong>only the lesion</strong> + a small margin of healthy skin<br>
        • Avoid hair, clothing, or bright reflections<br>
        • Skip this step if the whole image shows the affected area
        </div>
        """, unsafe_allow_html=True)
        want_crop = st.toggle(
            "✂️ Crop the image before analysis",
            value=False,
            key=f"want_crop_{file_id}",
        )

    if want_crop:
        st.markdown("#### ✂️ Crop to the Disease Area")
        st.caption("Drag the purple handles to select the region, then continue below.")
        crop_col, prev_col = st.columns([2, 1])
        with crop_col:
            MAX_CROP_W = 600
            if original_image.width > MAX_CROP_W:
                scale       = MAX_CROP_W / original_image.width
                crop_h      = int(original_image.height * scale)
                display_img = original_image.resize((MAX_CROP_W, crop_h), Image.LANCZOS)
            else:
                display_img = original_image
            cropped_image = st_cropper(
                display_img,
                realtime_update=True,
                box_color="#6366f1",
                aspect_ratio=None,
                return_type="image",
                key=f"skin_cropper_{file_id}",
            )
        with prev_col:
            if cropped_image is not None:
                st.markdown("**Crop preview:**")
                st.image(cropped_image, use_container_width=True)
                w, h = cropped_image.size
                st.caption(f"{w} × {h} px")
        if cropped_image is not None and cropped_image.size[0] > 10 and cropped_image.size[1] > 10:
            image = cropped_image
            st.success("✅ Cropped image will be used for AI analysis.")
        else:
            image = original_image
            st.warning("⚠️ Crop too small — using the full image.")
    else:
        image = original_image

    st.markdown("---")

    # ── Inference ──
    with st.spinner("Running AI differential diagnosis…"):
        try:
            model = load_model(model_path)
            predicted_disease, confidence, all_probs, ood_flag, mc_variance, low_conf, high_var = predict_disease(model, image)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

    st.subheader("🔬 Predicted Diagnosis")

    if ood_flag:
        reason = []
        if low_conf:
            reason.append(f"low confidence ({confidence*100:.1f}% < {OOD_CONF_THRESHOLD*100:.0f}%)")
        if high_var:
            reason.append(f"high prediction variance ({mc_variance:.4f} > {OOD_VAR_THRESHOLD})")
        reason_str = " and ".join(reason)

        st.markdown("""
        <div style="
            background: rgba(239,68,68,0.08);
            border: 1px solid rgba(239,68,68,0.25);
            border-left: 4px solid #ef4444;
            border-radius: 0 12px 12px 0;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.75rem;
        ">
        <span style="font-size:1.15rem;font-weight:700;color:#ef4444;">🚫 Unseen / Unknown Image</span><br><br>
        <span style="font-size:0.88rem;line-height:1.7;">
        This image does not match any of the <strong>32 disease classes</strong> the model was trained on.<br>
        The model's OOD detector flagged it due to <strong>{reason_str}</strong>.<br><br>
        This can happen when:<br>
        &nbsp;&nbsp;• The image is not a skin condition (e.g. a normal photo, X-ray, object)<br>
        &nbsp;&nbsp;• The condition is outside the model's training scope<br>
        &nbsp;&nbsp;• Image quality is very poor or the lesion area is not visible<br><br>
        <strong>Please consult a qualified dermatologist for evaluation.</strong>
        </span>
        </div>
        """.format(reason_str=reason_str), unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Model Confidence", f"{confidence*100:.1f}%",
                      delta=f"Threshold: {OOD_CONF_THRESHOLD*100:.0f}%", delta_color="inverse")
        col_m2.metric("Prediction Variance", f"{mc_variance:.4f}",
                      delta=f"Threshold: {OOD_VAR_THRESHOLD}", delta_color="inverse")

        st.markdown("**Closest matches (unreliable — for reference only):**")
        top5_idx = np.argsort(all_probs)[::-1][:5]
        for idx in top5_idx:
            st.markdown(f"- {DISEASE_CLASSES[idx]}: `{all_probs[idx]*100:.1f}%`")

        st.stop()

    else:
        diag_col, metric_col = st.columns([2, 1])
        with diag_col:
            st.success(f"✅ **Primary diagnosis: {predicted_disease}**")
        with metric_col:
            st.metric("Model Confidence", f"{confidence*100:.1f}%")
            st.metric("MC Variance", f"{mc_variance:.4f}", help="Lower = more certain")

        top5_idx = np.argsort(all_probs)[::-1][:5]
        st.markdown("**Top-5 differential diagnoses:**")
        for idx in top5_idx:
            st.markdown(f"- {DISEASE_CLASSES[idx]}: `{all_probs[idx]*100:.1f}%`")

    st.divider()

    # ── Step 2: Clinical assessment form ──
    st.header("📋 Step 2 — Clinical Assessment & Location")

    with st.form("clinical_assessment"):
        symptoms_data, personal_data, _ = render_disease_specific_form(
            predicted_disease, st.container()
        )
        submitted = st.form_submit_button(
            "🔬 Generate Personalised Recommendations", type="primary"
        )

    if not submitted:
        st.stop()

    city = personal_data.get('city', '').strip()
    with st.spinner(f"Fetching live weather & AQI for '{city or 'unknown location'}'…"):
        weather = get_weather_with_aqi(city, api_key.strip())

    severity_score = compute_severity_score(predicted_disease, symptoms_data, personal_data, weather)
    severity_label = score_to_severity(severity_score)

    recs = get_professional_recommendations(
        predicted_disease, severity_label, weather,
        personal_data.get('skin_type', 'Normal'),
        personal_data.get('allergies', []),
        existing_conditions=personal_data.get('existing_conditions', ''),
        immune_detail=personal_data.get('immune_detail', '')
    )

    st.divider()

    severity_color_map = {"Mild": "#22c55e", "Moderate": "#f59e0b", "Severe": "#ef4444"}
    sev_color = severity_color_map.get(severity_label, "#6366f1")

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
    .results-header {{font-family:'DM Serif Display',serif;font-size:2rem;font-weight:400;letter-spacing:-0.02em;margin-bottom:0.25rem;color:inherit;}}
    .results-subheader {{font-family:'DM Sans',sans-serif;font-size:0.9rem;opacity:0.6;margin-bottom:1.5rem;font-weight:400;}}
    .weather-badge {{display:inline-flex;align-items:center;gap:0.5rem;padding:0.45rem 1rem;border-radius:999px;font-family:'DM Sans',sans-serif;font-size:0.82rem;font-weight:500;background:rgba(34,197,94,0.12);color:#22c55e;border:1px solid rgba(34,197,94,0.25);margin-bottom:1.5rem;}}
    .weather-badge-warn {{background:rgba(245,158,11,0.12);color:#f59e0b;border-color:rgba(245,158,11,0.25);}}
    .metrics-grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.25rem;}}
    .metric-card {{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1.1rem 1.25rem;position:relative;overflow:hidden;}}
    .metric-card::before {{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),transparent);border-radius:14px 14px 0 0;}}
    .metric-label {{font-family:'DM Sans',sans-serif;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;opacity:0.5;margin-bottom:0.35rem;}}
    .metric-value {{font-family:'DM Serif Display',serif;font-size:1.6rem;font-weight:400;line-height:1.15;letter-spacing:-0.02em;}}
    .metric-value.severity-mild {{color:#22c55e;}}
    .metric-value.severity-moderate {{color:#f59e0b;}}
    .metric-value.severity-severe {{color:#ef4444;}}
    .metrics-row-2 {{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.25rem;}}
    .metric-card-sm {{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.85rem 1rem;}}
    .metric-label-sm {{font-family:'DM Sans',sans-serif;font-size:0.68rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;opacity:0.45;margin-bottom:0.25rem;}}
    .metric-value-sm {{font-family:'DM Sans',sans-serif;font-size:1.15rem;font-weight:600;}}
    .allergy-bar {{background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-left:3px solid #6366f1;border-radius:0 10px 10px 0;padding:0.75rem 1.1rem;font-family:'DM Sans',sans-serif;font-size:0.85rem;margin-bottom:1rem;}}
    .warning-card {{background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-left:3px solid #ef4444;border-radius:0 10px 10px 0;padding:0.85rem 1.1rem;font-family:'DM Sans',sans-serif;font-size:0.85rem;margin-bottom:0.6rem;line-height:1.6;}}
    .rec-section-header {{font-family:'DM Sans',sans-serif;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;opacity:0.45;margin:1.75rem 0 0.75rem 0;display:flex;align-items:center;gap:0.5rem;}}
    .rec-section-header::after {{content:'';flex:1;height:1px;background:rgba(255,255,255,0.07);}}
    .rec-cards-grid {{display:flex;flex-direction:column;gap:0.45rem;}}
    .rec-item {{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:0.7rem 0.9rem;font-family:'DM Sans',sans-serif;font-size:0.82rem;line-height:1.65;display:flex;gap:0.55rem;align-items:flex-start;}}
    .rec-item .dot {{width:6px;height:6px;border-radius:50%;margin-top:0.48rem;flex-shrink:0;}}
    .disclaimer-box {{background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.18);border-radius:12px;padding:1rem 1.25rem;font-family:'DM Sans',sans-serif;font-size:0.8rem;line-height:1.65;margin-top:2rem;opacity:0.85;}}
    .disclaimer-box strong {{font-weight:700;letter-spacing:0.03em;}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="results-header">📊 Your Personalised Skin Care Plan</p>', unsafe_allow_html=True)
    st.markdown('<p class="results-subheader">Written in plain language • Based on your symptoms, location & health profile</p>', unsafe_allow_html=True)

    src = weather.get("source", "mock")
    if src == "live":
        st.markdown(
            f'<div class="weather-badge">🌐 Live data · {weather["city"]} · {weather["description"].title()}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="weather-badge weather-badge-warn">⚠️ Simulated weather data — enter a valid city for live data</div>',
            unsafe_allow_html=True
        )

    sev_class = f"severity-{severity_label.lower()}"
    st.markdown(f"""
    <div class="metrics-grid" style="--accent:{sev_color};">
        <div class="metric-card"><div class="metric-label">🔬 Diagnosis</div><div class="metric-value">{recs["personalization"]["disease"]}</div></div>
        <div class="metric-card" style="--accent:{sev_color};"><div class="metric-label">📊 Severity</div><div class="metric-value {sev_class}">{recs["personalization"]["severity"]}</div></div>
        <div class="metric-card" style="--accent:#6366f1;"><div class="metric-label">📈 Severity Score</div><div class="metric-value">{severity_score}<span style="font-family:'DM Sans';font-size:1rem;opacity:0.4;">/100</span></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row-2">
        <div class="metric-card-sm"><div class="metric-label-sm">🌡️ Temperature</div><div class="metric-value-sm">{recs["personalization"]["temperature"]}</div></div>
        <div class="metric-card-sm"><div class="metric-label-sm">💧 Humidity</div><div class="metric-value-sm">{recs["personalization"]["humidity"]}</div></div>
        <div class="metric-card-sm"><div class="metric-label-sm">🌫️ Air Quality</div><div class="metric-value-sm">{recs["personalization"]["aqi"].upper()}</div></div>
        <div class="metric-card-sm"><div class="metric-label-sm">🎨 Skin Type</div><div class="metric-value-sm">{recs["personalization"]["skin_type"]}</div></div>
    </div>
    """, unsafe_allow_html=True)

    allergy_str = ', '.join(recs['personalization']['allergies'])
    st.markdown(f'<div class="allergy-bar"><strong>Documented Allergies:</strong> {allergy_str}</div>', unsafe_allow_html=True)
    if recs["personalization"]["other_conditions"] != "None reported":
        st.markdown(f'<div class="allergy-bar"><strong>Other Health Conditions:</strong> {recs["personalization"]["other_conditions"]}</div>', unsafe_allow_html=True)

    if recs.get("warnings"):
        st.markdown('<div class="rec-section-header">⚠️ Important Warnings</div>', unsafe_allow_html=True)
        for w in recs["warnings"]:
            st.markdown(f'<div class="warning-card">{w}</div>', unsafe_allow_html=True)

    sections = [
        ("🏥", "What to do right now",          recs.get("immediate_care", []),                "#6366f1"),
        ("💊", "Medicines & treatments",          recs.get("medications", []),                    "#ec4899"),
        ("🛡️", "Protecting your skin daily",     recs.get("environmental_precautions", []),      "#06b6d4"),
        ("🌿", "Healthy habits that help",        recs.get("lifestyle_adjustments", []),          "#22c55e"),
        ("⏰", "When to see a doctor",            recs.get("when_to_seek_medical_attention", []), "#f59e0b"),
    ]
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    for (icon, title, items, color) in sections:
        st.markdown(
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:1.1rem;font-weight:700;'
            f'letter-spacing:0em;text-transform:uppercase;opacity:0.6;margin-bottom:0.7rem;'
            f'border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:0.5rem;">'
            f'{icon} {title}</div>',
            unsafe_allow_html=True
        )
        if items:
            items_html = "".join(
                f'<div class="rec-item"><span class="dot" style="background:{color};opacity:0.85;"></span>'
                f'<span>{item}</span></div>'
                for item in items
            )
            st.markdown(f'<div class="rec-cards-grid">{items_html}</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom:1.25rem;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ IMPORTANT:</strong> This app gives you helpful guidance, but it is <strong>not a replacement for a real doctor</strong>.
        Please visit a qualified skin doctor (dermatologist) or your family doctor to confirm your diagnosis and treatment.
        Do not start or stop any medication based only on this app.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()