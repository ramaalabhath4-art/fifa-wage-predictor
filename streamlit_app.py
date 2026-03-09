import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(ROOT_DIR, "models", "best_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "features.json")

# ── Wage imputation table (median wage per Overall bracket from training data) ─
# This fixes the Wage=0 → always-Low bug.
# When the user leaves Wage=0 (unknown), we substitute the typical wage for
# that Overall rating so the model receives a realistic value.
WAGE_BY_OVERALL = [
    (91, 99,  380),
    (88, 91,  215),
    (85, 88,  148),
    (82, 85,   72),
    (79, 82,   42),
    (76, 79,   26),
    (73, 76,   17),
    (70, 73,    9),
    (65, 70,    3),
    (60, 65,    2),
    ( 0, 60,    1),
]

def impute_wage(overall: int) -> int:
    """Return the typical (median) wage for a given Overall rating."""
    for lo, hi, med in WAGE_BY_OVERALL:
        if lo <= overall < hi:
            return med
    return 1


# ── All 651 clubs from the training dataset ───────────────────────────────────
CLUBS = [
    ' SSV Jahn Regensburg', '1. FC Heidenheim 1846', '1. FC Kaiserslautern',
    '1. FC Köln', '1. FC Magdeburg', '1. FC Nürnberg', '1. FC Union Berlin',
    '1. FSV Mainz 05', 'AC Ajaccio', 'AC Horsens', 'AD Alcorcón', 'ADO Den Haag',
    'AEK Athens', 'AFC Wimbledon', 'AIK', 'AJ Auxerre', 'AS Béziers', 'AS Monaco',
    'AS Nancy Lorraine', 'AS Saint-Étienne', 'AZ Alkmaar', 'Aalborg BK', 'Aarhus GF',
    'Aberdeen', 'Accrington Stanley', 'Adelaide United', 'Ajax', 'Akhisar Belediyespor',
    'Al Ahli', 'Al Batin', 'Al Faisaly', 'Al Fateh', 'Al Fayha', 'Al Hazem',
    'Al Hilal', 'Al Ittihad', 'Al Nassr', 'Al Qadisiyah', 'Al Raed', 'Al Shabab',
    'Al Taawoun', 'Al Wehda', 'Alanyaspor', 'Albacete BP', 'Alianza Petrolera',
    'Amiens SC', 'América FC (Minas Gerais)', 'América de Cali', 'Angers SCO',
    'Antalyaspor', 'Argentinos Juniors', 'Arka Gdynia', 'Arsenal', 'Ascoli',
    'Aston Villa', 'Atalanta', 'Athletic Club de Bilbao', 'Atiker Konyaspor',
    'Atlanta United', 'Atlético Bucaramanga', 'Atlético Huila', 'Atlético Madrid',
    'Atlético Mineiro', 'Atlético Nacional', 'Atlético Paranaense', 'Atlético Tucumán',
    'Audax Italiano', 'BB Erzurumspor', 'BK Häcken', 'BSC Young Boys', 'Bahia',
    'Barnsley', 'Bayer 04 Leverkusen', 'Beijing Renhe FC', 'Beijing Sinobo Guoan FC',
    'Belgrano de Córdoba', 'Benevento', 'Beşiktaş JK', 'Birmingham City',
    'Blackburn Rovers', 'Blackpool', 'Boavista FC', 'Boca Juniors', 'Bohemian FC',
    'Bologna', 'Bolton Wanderers', 'Borussia Dortmund', 'Borussia Mönchengladbach',
    'Botafogo', 'Bournemouth', 'Boyacá Chicó FC', 'Bradford City', 'Bray Wanderers',
    'Brentford', 'Brescia', 'Brighton & Hove Albion', 'Brisbane Roar', 'Bristol City',
    'Bristol Rovers', 'Brøndby IF', 'Burnley', 'Bursaspor', 'Burton Albion', 'Bury',
    'CA Osasuna', 'CD Antofagasta', 'CD Aves', 'CD Everton de Viña del Mar',
    'CD Feirense', 'CD Huachipato', 'CD Leganés', 'CD Lugo', 'CD Nacional',
    'CD Numancia', "CD O'Higgins", 'CD Palestino', 'CD Tenerife', 'CD Tondela',
    'CD Universidad de Concepción', 'CF Rayo Majadahonda', 'CF Reus Deportiu',
    'Cagliari', 'Cambridge United', 'Cardiff City', 'Carlisle United', 'Carpi',
    'Ceará Sporting Club', 'Celtic', 'Central Coast Mariners', 'Cerezo Osaka',
    'Chamois Niortais Football Club', 'Changchun Yatai FC', 'Chapecoense',
    'Charlton Athletic', 'Chelsea', 'Cheltenham Town', 'Chicago Fire', 'Chievo Verona',
    'Chongqing Dangdai Lifan FC SWM Team', 'Cittadella', 'Clermont Foot 63',
    'Club América', 'Club Atlas', 'Club Atlético Aldosivi', 'Club Atlético Banfield',
    'Club Atlético Colón', 'Club Atlético Huracán', 'Club Atlético Lanús',
    'Club Atlético Talleres', 'Club Atlético Tigre', 'Club Brugge KV',
    'Club Deportes Temuco', 'Club León', 'Club Necaxa', 'Club Tijuana',
    'Clube Sport Marítimo', 'Colchester United', 'Colo-Colo', 'Colorado Rapids',
    'Columbus Crew SC', 'Cork City', 'Cosenza', 'Coventry City', 'Cracovia',
    'Crawley Town', 'Crewe Alexandra', 'Crotone', 'Cruz Azul', 'Cruzeiro',
    'Crystal Palace', 'Curicó Unido', 'Cádiz CF', 'Córdoba CF', 'DC United',
    'DSC Arminia Bielefeld', 'Daegu FC', 'Dalian YiFang FC', 'Dalkurd FF',
    'De Graafschap', 'Defensa y Justicia', 'Deportes Iquique', 'Deportes Tolima',
    'Deportivo Alavés', 'Deportivo Cali', 'Deportivo Pasto', 'Deportivo Toluca',
    'Deportivo de La Coruña', 'Derby County', 'Derry City', 'Dijon FCO',
    'Dinamo Zagreb', 'Djurgårdens IF', 'Doncaster Rovers', 'Dundalk', 'Dundee FC',
    'Dynamo Kyiv', 'ESTAC Troyes', 'Eintracht Braunschweig', 'Eintracht Frankfurt',
    'Elche CF', 'Empoli', 'En Avant de Guingamp', 'Envigado FC', 'Esbjerg fB',
    'Estudiantes de La Plata', 'Ettifaq FC', 'Everton', 'Excelsior', 'Exeter City',
    'Extremadura UD', 'FC Admira Wacker Mödling', 'FC Augsburg', 'FC Barcelona',
    'FC Basel 1893', 'FC Bayern München', 'FC Carl Zeiss Jena', 'FC Dallas',
    'FC Emmen', 'FC Energie Cottbus', 'FC Erzgebirge Aue', 'FC Girondins de Bordeaux',
    'FC Groningen', 'FC Hansa Rostock', 'FC Ingolstadt 04', 'FC København',
    'FC Lorient', 'FC Lugano', 'FC Luzern', 'FC Metz', 'FC Midtjylland', 'FC Nantes',
    'FC Nordsjælland', 'FC Porto', 'FC Red Bull Salzburg', 'FC Schalke 04', 'FC Seoul',
    'FC Sion', 'FC Sochaux-Montbéliard', 'FC St. Gallen', 'FC St. Pauli', 'FC Thun',
    'FC Tokyo', 'FC Utrecht', 'FC Wacker Innsbruck', 'FC Würzburger Kickers',
    'FC Zürich', 'FK Austria Wien', 'FK Bodø/Glimt', 'FK Haugesund', 'FSV Zwickau',
    'Fenerbahçe SK', 'Feyenoord', 'Fiorentina', 'Fleetwood Town', 'Fluminense',
    'Foggia', 'Forest Green Rovers', 'Fortuna Düsseldorf', 'Fortuna Sittard',
    'Frosinone', 'Fulham', 'GD Chaves', 'GFC Ajaccio', 'GIF Sundsvall',
    'Galatasaray SK', 'Gamba Osaka', 'Gangwon FC', 'Genoa', 'Getafe CF', 'Gillingham',
    'Gimnasia y Esgrima La Plata', 'Gimnàstic de Tarragona', 'Girona FC', 'Godoy Cruz',
    'Granada CF', 'Grasshopper Club Zürich', 'Grenoble Foot 38', 'Grimsby Town',
    'Grêmio', 'Guadalajara', 'Guangzhou Evergrande Taobao FC', 'Guangzhou R&F; FC',
    'Guizhou Hengfeng FC', 'Gyeongnam FC', 'Górnik Zabrze', 'Göztepe SK',
    'HJK Helsinki', 'Hallescher FC', 'Hamburger SV', 'Hamilton Academical FC',
    'Hammarby IF', 'Hannover 96', 'Heart of Midlothian', 'Hebei China Fortune FC',
    'Hellas Verona', 'Henan Jianye FC', 'Heracles Almelo', 'Hertha BSC', 'Hibernian',
    'Hobro IK', 'Hokkaido Consadole Sapporo', 'Holstein Kiel', 'Houston Dynamo',
    'Huddersfield Town', 'Hull City', 'IF Brommapojkarna', 'IF Elfsborg',
    'IFK Göteborg', 'IFK Norrköping', 'IK Sirius', 'IK Start', 'Incheon United FC',
    'Independiente', 'Independiente Medellín', 'Independiente Santa Fe', 'Inter',
    'Internacional', 'Ipswich Town', 'Itagüí Leones FC', 'Jagiellonia Białystok',
    'Jaguares de Córdoba', 'Jeju United FC', 'Jeonbuk Hyundai Motors',
    'Jeonnam Dragons', 'Jiangsu Suning FC', 'Junior FC', 'Juventus', 'Júbilo Iwata',
    'KAA Gent', 'KAS Eupen', 'KFC Uerdingen 05', 'KRC Genk', 'KSV Cercle Brugge',
    'KV Kortrijk', 'KV Oostende', 'Kaizer Chiefs', 'Kalmar FF', 'Karlsruher SC',
    'Kashima Antlers', 'Kashiwa Reysol', 'Kasimpaşa SK', 'Kawasaki Frontale',
    'Kayserispor', 'Kilmarnock', 'Korona Kielce', 'Kristiansund BK', 'LA Galaxy',
    'LASK Linz', 'LOSC Lille', 'La Berrichonne de Châteauroux', 'La Equidad', 'Lazio',
    'Le Havre AC', 'Lecce', 'Lech Poznań', 'Lechia Gdańsk', 'Leeds United',
    'Legia Warszawa', 'Leicester City', 'Levante UD', 'Lillestrøm SK', 'Limerick FC',
    'Lincoln City', 'Liverpool', 'Livingston FC', 'Livorno', 'Lobos BUAP',
    'Lokomotiv Moscow', 'Los Angeles FC', 'Luton Town', 'MKE Ankaragücü',
    'MSV Duisburg', 'Macclesfield Town', 'Malmö FF', 'Manchester City',
    'Manchester United', 'Mansfield Town', 'Medipol Başakşehir FK',
    'Melbourne City FC', 'Melbourne Victory', 'Middlesbrough', 'Miedź Legnica',
    'Milan', 'Millonarios FC', 'Millwall', 'Milton Keynes Dons', 'Minnesota United FC',
    'Molde FK', 'Monarcas Morelia', 'Monterrey', 'Montpellier HSC', 'Montreal Impact',
    'Morecambe', 'Moreirense FC', 'Motherwell', 'Málaga CF', 'NAC Breda',
    'Nagoya Grampus', 'Napoli', 'Neuchâtel Xamax', 'New England Revolution',
    'New York City FC', 'New York Red Bulls', 'Newcastle Jets', 'Newcastle United',
    "Newell's Old Boys", 'Newport County', 'Northampton Town', 'Norwich City',
    'Nottingham Forest', 'Notts County', 'Nîmes Olympique', 'OGC Nice', 'Odds BK',
    'Odense Boldklub', 'Ohod Club', 'Oldham Athletic', 'Olympiacos CFP',
    'Olympique Lyonnais', 'Olympique de Marseille', 'Once Caldas', 'Orlando City SC',
    'Orlando Pirates', 'Os Belenenses', 'Oxford United', 'PAOK', 'PEC Zwolle',
    'PFC CSKA Moscow', 'PSV', 'Pachuca', 'Padova', 'Palermo', 'Panathinaikos FC',
    'Paraná', 'Paris FC', 'Paris Saint-Germain', 'Parma', 'Patriotas Boyacá FC',
    'Patronato', 'Perth Glory', 'Perugia', 'Pescara', 'Peterborough United',
    'Philadelphia Union', 'Piast Gliwice', 'Plymouth Argyle', 'Pogoń Szczecin',
    'Pohang Steelers', 'Port Vale', 'Portimonense SC', 'Portland Timbers',
    'Portsmouth', 'Preston North End', 'Puebla FC', 'Queens Park Rangers', 'Querétaro',
    'RB Leipzig', 'RC Celta', 'RC Strasbourg Alsace', 'RCD Espanyol', 'RCD Mallorca',
    'RSC Anderlecht', 'Racing Club', 'Racing Club de Lens', 'Randers FC', 'Rangers FC',
    'Ranheim Fotball', 'Rayo Vallecano', 'Reading', 'Real Betis', 'Real Madrid',
    'Real Oviedo', 'Real Salt Lake', 'Real Sociedad', 'Real Sporting de Gijón',
    'Real Valladolid CF', 'Real Zaragoza', 'Red Star FC', 'Rio Ave FC',
    'Rionegro Águilas', 'River Plate', 'Rochdale', 'Roma', 'Rosario Central',
    'Rosenborg BK', 'Rotherham United', 'Royal Antwerp FC', 'Royal Excel Mouscron',
    'SC Braga', 'SC Fortuna Köln', 'SC Freiburg', 'SC Heerenveen', 'SC Paderborn 07',
    'SC Preußen Münster', 'SCR Altach', 'SD Eibar', 'SD Huesca', 'SG Dynamo Dresden',
    'SG Sonnenhof Großaspach', 'SK Brann', 'SK Rapid Wien', 'SK Slavia Praha',
    'SK Sturm Graz', 'SKN St. Pölten', 'SL Benfica', 'SPAL', 'SV Darmstadt 98',
    'SV Mattersburg', 'SV Meppen', 'SV Sandhausen', 'SV Wehen Wiesbaden',
    'SV Werder Bremen', 'SV Zulte-Waregem', 'Sagan Tosu', 'Sampdoria',
    'San Jose Earthquakes', 'San Lorenzo de Almagro', 'San Luis de Quillota',
    'San Martin de Tucumán', 'San Martín de San Juan', 'Sandefjord Fotball',
    'Sanfrecce Hiroshima', 'Sangju Sangmu FC', 'Santa Clara', 'Santos',
    'Santos Laguna', 'Sarpsborg 08 FF', 'Sassuolo', 'Scunthorpe United',
    'Seattle Sounders FC', 'Sevilla FC', 'Shakhtar Donetsk', 'Shamrock Rovers',
    'Shandong Luneng TaiShan FC', 'Shanghai Greenland Shenhua FC', 'Shanghai SIPG FC',
    'Sheffield United', 'Sheffield Wednesday', 'Shimizu S-Pulse', 'Shonan Bellmare',
    'Shrewsbury', 'Sint-Truidense VV', 'Sivasspor', 'Sligo Rovers', 'Southampton',
    'Southend United', 'SpVgg Greuther Fürth', 'SpVgg Unterhaching', 'Sparta Praha',
    'Spartak Moscow', 'Spezia', 'Sport Club do Recife', 'Sporting CP',
    'Sporting Kansas City', 'Sporting Lokeren', 'Sporting de Charleroi',
    'St. Johnstone FC', 'St. Mirren', "St. Patrick's Athletic", 'Stabæk Fotball',
    'Stade Brestois 29', 'Stade Malherbe Caen', 'Stade Rennais FC', 'Stade de Reims',
    'Standard de Liège', 'Stevenage', 'Stoke City', 'Strømsgodset IF', 'Sunderland',
    'Suwon Samsung Bluewings', 'Swansea City', 'Swindon Town', 'Sydney FC',
    'SønderjyskE', 'TSG 1899 Hoffenheim', 'TSV 1860 München', 'TSV Hartberg',
    'Tianjin Quanjian FC', 'Tianjin TEDA FC', 'Tiburones Rojos de Veracruz',
    'Tigres U.A.N.L.', 'Torino', 'Toronto FC', 'Tottenham Hotspur',
    'Toulouse Football Club', 'Trabzonspor', 'Tranmere Rovers', 'Trelleborgs FF',
    'Tromsø IL', 'U.N.A.M.', 'UD Almería', 'UD Las Palmas', 'US Cremonese',
    'US Orléans Loiret Football', 'US Salernitana 1919', 'Udinese', 'Ulsan Hyundai FC',
    'Universidad Católica', 'Universidad de Chile', 'Unión Española', 'Unión La Calera',
    'Unión de Santa Fe', 'Urawa Red Diamonds', 'V-Varen Nagasaki', 'VVV-Venlo',
    'Valencia CF', 'Valenciennes FC', 'Vancouver Whitecaps FC', 'Vegalta Sendai',
    'Vejle Boldklub', 'Vendsyssel FF', 'Venezia FC', 'VfB Stuttgart', 'VfL Bochum 1848',
    'VfL Osnabrück', 'VfL Sportfreunde Lotte', 'VfL Wolfsburg', 'VfR Aalen',
    'Viktoria Plzeň', 'Villarreal CF', 'Vissel Kobe', 'Vitesse', 'Vitória',
    'Vitória Guimarães', 'Vitória de Setúbal', 'Vålerenga Fotball', 'Vélez Sarsfield',
    'Waasland-Beveren', 'Walsall', 'Waterford FC', 'Watford', 'Wellington Phoenix',
    'West Bromwich Albion', 'West Ham United', 'Western Sydney Wanderers',
    'Wigan Athletic', 'Willem II', 'Wisła Kraków', 'Wisła Płock', 'Wolfsberger AC',
    'Wolverhampton Wanderers', 'Wycombe Wanderers', 'Yeni Malatyaspor', 'Yeovil Town',
    'Yokohama F. Marinos', 'Zagłębie Lubin', 'Zagłębie Sosnowiec', 'Çaykur Rizespor',
    'Örebro SK', 'Östersunds FK', 'Śląsk Wrocław',
]

POSITIONS = [
    'CAM','CB','CDM','CF','CM','GK','LAM','LB','LCB','LCM','LDM',
    'LF','LM','LS','LW','LWB','RAM','RB','RCB','RCM','RDM',
    'RF','RM','RS','RW','RWB','ST',
]

WAGE_TIERS = [
    (0,    10,   "Low",        "🟤", "#a0522d"),
    (10,   30,   "Below Avg",  "🔵", "#4682b4"),
    (30,   80,   "Average",    "🟢", "#2e8b57"),
    (80,   200,  "Above Avg",  "🟡", "#daa520"),
    (200,  500,  "High",       "🟠", "#ff8c00"),
    (500,  9999, "Elite",      "🔴", "#dc143c"),
]

def get_tier(wage):
    for lo, hi, label, icon, color in WAGE_TIERS:
        if lo <= wage < hi:
            return label, icon, color
    return "Elite", "🔴", "#dc143c"

def tier_probabilities(pred_wage, sigma_log=0.18):
    rng      = np.random.default_rng(42)
    pred_log = np.log1p(pred_wage)
    samples  = np.expm1(rng.normal(pred_log, sigma_log, size=20_000))
    probs    = {}
    for lo, hi, label, _, _ in WAGE_TIERS:
        count        = np.sum((samples >= lo) & (samples < hi))
        probs[label] = round(float(count) / len(samples) * 100, 1)
    return probs


@st.cache_resource
def load_model_and_metadata():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model not found at `{MODEL_PATH}`\n\n"
            "Run `python src/train.py` first to generate it."
        )
        st.stop()
    model    = joblib.load(MODEL_PATH)
    metadata = json.load(open(METADATA_PATH, encoding="utf-8"))
    return model, metadata


def predict_player(model, player: dict) -> dict:
    # ── KEY FIX: if Wage is 0 (unknown), replace with typical wage for Overall ──
    if player["Wage"] == 0:
        player = {**player, "Wage": impute_wage(int(player["Overall"]))}

    df_input  = pd.DataFrame([player])
    pred_log  = model.predict(df_input)[0]
    pred_wage = np.expm1(pred_log)
    label, icon, color = get_tier(pred_wage)
    probs = tier_probabilities(pred_wage)
    return {
        "predicted_log_wage":    round(pred_log,  4),
        "predicted_weekly_wage": round(pred_wage, 2),
        "tier_label": label,
        "tier_icon":  icon,
        "tier_color": color,
        "tier_probs": probs,
        "wage_used":  player["Wage"],
    }


def main():
    st.set_page_config(page_title="FIFA Wage Predictor", page_icon="⚽", layout="wide")
    st.title("⚽ FIFA Wage Predictor")
    st.caption("Club and Position are full dropdowns from the training data. Leave Known Wage = 0 to auto-estimate it from Overall rating.")

    model, metadata = load_model_and_metadata()

    st.header("Single Player Prediction")

    with st.form("player_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            overall    = st.number_input("Overall",   min_value=46, max_value=99, value=80)
            potential  = st.number_input("Potential", min_value=48, max_value=99, value=85)
            age        = st.number_input("Age",       min_value=15, max_value=45, value=24)

        with c2:
            value      = st.number_input("Value (€K)",               min_value=0, value=30_000, step=500)
            release    = st.number_input("Release Clause (€K)",      min_value=0, value=60_000, step=500)
            wage_input = st.number_input(
                "Known Wage — leave 0 to auto-estimate",
                min_value=0, value=0, step=1,
                help="If you don't know the wage, leave this at 0. The app will automatically use the typical wage for this Overall rating."
            )

        with c3:
            club      = st.selectbox("Club",           options=CLUBS,     index=CLUBS.index("Arsenal"))
            position  = st.selectbox("Position",       options=POSITIONS, index=POSITIONS.index("CM"))
            foot      = st.selectbox("Preferred Foot", options=["Right", "Left"])

        submitted = st.form_submit_button("🔍 Predict Wage", use_container_width=True)

    if submitted:
        player = {
            "Overall":        overall,
            "Potential":      potential,
            "Value":          value,
            "Release Clause": release,
            "Age":            age,
            "Wage":           wage_input,
            "Position":       position,
            "Preferred Foot": foot,
            "Club":           club,
        }
        result = predict_player(model, player)
        wage   = result["predicted_weekly_wage"]

        st.markdown("---")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.metric("Predicted Weekly Wage", f"€{wage:,.0f}K")
            st.markdown(
                f"<div style='margin-top:8px'>"
                f"<span style='font-size:1.3rem;font-weight:700;color:{result['tier_color']}'>"
                f"{result['tier_icon']} {result['tier_label']} tier</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Log-wage: {result['predicted_log_wage']}")
            st.caption(f"Wage used for model: {result['wage_used']}K")
            st.caption(f"Model: {metadata['best_model_name']}  |  R² = {metadata['best_r2']:.4f}")

        with col_b:
            st.subheader("Wage-Tier Probabilities")
            for _, _, label, icon, color in WAGE_TIERS:
                p = result["tier_probs"].get(label, 0)
                if p == 0:
                    continue
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:7px'>"
                    f"<span style='width:105px;font-weight:600;color:{color}'>{icon} {label}</span>"
                    f"<div style='flex:1;background:#1a1d27;border-radius:6px;height:16px'>"
                    f"<div style='width:{max(int(p*3),3)}%;background:{color};height:100%;border-radius:6px'></div></div>"
                    f"<span style='width:45px;text-align:right;font-weight:700;color:{color}'>{p}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.header("Batch Prediction")
    st.caption("Required columns: Overall, Potential, Value, Release Clause, Age, Wage, Position, Preferred Foot, Club — set Wage=0 for unknown players")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df       = pd.read_csv(uploaded)
        features = metadata["selected_features"]
        missing  = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            # Auto-impute Wage=0 rows in batch too
            df["Wage"] = df.apply(
                lambda r: impute_wage(int(r["Overall"])) if r["Wage"] == 0 else r["Wage"],
                axis=1
            )
            preds_log  = model.predict(df[features])
            preds_wage = np.expm1(preds_log)
            df["Predicted_Weekly_Wage"] = preds_wage.round(2)
            df["Wage_Tier"] = df["Predicted_Weekly_Wage"].apply(lambda w: get_tier(w)[0])
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                "⬇ Download predictions",
                df.to_csv(index=False).encode("utf-8"),
                "predictions.csv", "text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()