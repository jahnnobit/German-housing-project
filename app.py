import streamlit as st
import pandas as pd
import joblib
import os
import base64
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="DE Real Estate Pro", layout="wide")

# --- 1. FUNCTION TO CONVERT IMAGE TO BASE64 (Required for Background) ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- 2. AUTO-TRAIN LOGIC ---
def initial_setup():
    if not os.path.exists('germany_pro_model.pkl') or not os.path.exists('state_city_map.pkl'):
        with st.status("Training the AI model...", expanded=True) as status:
            df = pd.read_csv('immo_data.csv')
            cols = ['regio1', 'regio2', 'livingSpace', 'noRooms', 'hasKitchen', 'balcony', 'lift', 'baseRent']
            df = df[cols].dropna()
            df = df[(df['livingSpace'] > 10) & (df['livingSpace'] < 300)]
            df = df[(df['baseRent'] > 100) & (df['baseRent'] < 5000)]
            top_cities = df['regio2'].value_counts().nlargest(100).index
            df = df[df['regio2'].isin(top_cities)]
            state_city_map = df.groupby('regio1')['regio2'].unique().apply(list).to_dict()
            joblib.dump(state_city_map, 'state_city_map.pkl')
            avg_rent = df.groupby('regio1')['baseRent'].mean().reset_index()
            joblib.dump(avg_rent, 'avg_rent_data.pkl')
            df_encoded = pd.get_dummies(df, columns=['regio1', 'regio2'])
            X = df_encoded.drop('baseRent', axis=1)
            y = df_encoded['baseRent']
            model = LinearRegression()
            model.fit(X, y)
            joblib.dump(model, 'germany_pro_model.pkl')
            joblib.dump(X.columns.tolist(), 'pro_model_columns.pkl')
            status.update(label="Setup Complete!", state="complete", expanded=False)

initial_setup()

# --- 3. LOAD ASSETS ---
model = joblib.load('germany_pro_model.pkl')
model_cols = joblib.load('pro_model_columns.pkl')
state_city_map = joblib.load('state_city_map.pkl')
avg_rent_df = joblib.load('avg_rent_data.pkl')

# --- 4. NAVIGATION LOGIC ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def go_to(page_name):
    st.session_state.page = page_name

def get_prediction(state, city, size, rooms, kitchen, balcony, lift):
    input_df = pd.DataFrame(0, index=[0], columns=model_cols)
    input_df['livingSpace'] = size
    input_df['noRooms'] = rooms
    input_df['hasKitchen'] = 1 if kitchen else 0
    input_df['balcony'] = 1 if balcony else 0
    input_df['lift'] = 1 if lift else 0
    s_col, c_col = f"regio1_{state}", f"regio2_{city}"
    if s_col in model_cols: input_df[s_col] = 1
    if c_col in model_cols: input_df[c_col] = 1
    return model.predict(input_df)[0]

# --- 5. PAGE ROUTING ---

# 🏠 HOME PAGE
if st.session_state.page == 'Home':
    # Attempt to load your local background image
    try:
        bin_str = get_base64_image("website home page.jpg")
        bg_image_style = f"background-image: url('data:image/jpg;base64,{bin_str}');"
    except:
        # Fallback to a URL if file is missing
        bg_image_style = "background-image: url('https://images.unsplash.com/photo-1467226632440-65f0b4957563?q=80&w=1000');"

    # HTML/CSS for the Hero Header with Text Overlay
    st.markdown(f"""
        <div style="
            {bg_image_style}
            background-size: cover;
            background-position: center;
            height: 350px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            border-radius: 15px;
            margin-bottom: 30px;
            ">
            <h1 style="color: white; text-shadow: 2px 2px 8px #000000; font-size: 50px; margin-bottom: 0;"> DE Real Estate Pro</h1>
            <p style="color: white; text-shadow: 2px 2px 8px #000000; font-size: 20px;">Your AI-Powered Guide to the German Housing Market</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("### What would you like to do today?")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("### 🏙️ Compare Cities\nAnalyze two locations side-by-side.")
        if st.button("Start Comparing"): go_to('Compare')
    with c2:
        st.success("### 🏠 Single Inquiry\nGet an estimate for one property.")
        if st.button("Get Estimate"): go_to('Inquiry')
    with c3:
        st.warning("### ⚖️ Deal Evaluator\nCheck if a price is fair.")
        if st.button("Evaluate Deal"): go_to('Evaluator')
        
    st.write("---")
    st.plotly_chart(px.bar(avg_rent_df, x='regio1', y='baseRent', title="National Averages: Rent by State"), use_container_width=True)

# 🏙️ COMPARE PAGE
elif st.session_state.page == 'Compare':
    if st.sidebar.button("⬅️ Back to Home"): go_to('Home')
    st.title("City Comparison Mode")
    st.sidebar.header("Property Features")
    sqm = st.sidebar.number_input("Size (m²)", 10, 300, 80)
    rms = st.sidebar.slider("Rooms", 1.0, 6.0, 3.0)
    l, r = st.columns(2)
    with l:
        st.subheader("Location A")
        s1 = st.selectbox("State A", sorted(state_city_map.keys()), key="s1")
        city1 = st.selectbox("City A", sorted(state_city_map[s1]), key="c1")
        p1 = get_prediction(s1, city1, sqm, rms, 1, 1, 0)
        st.metric(f"Rent in {city1}", f"€{p1:,.2f}")
    with r:
        st.subheader("Location B")
        s2 = st.selectbox("State B", sorted(state_city_map.keys()), key="s2", index=1)
        city2 = st.selectbox("City B", sorted(state_city_map[s2]), key="c2")
        p2 = get_prediction(s2, city2, sqm, rms, 1, 1, 0)
        st.metric(f"Rent in {city2}", f"€{p2:,.2f}", delta=f"{p2-p1:,.2f}", delta_color="inverse")

# 🏠 INQUIRY PAGE
elif st.session_state.page == 'Inquiry':
    if st.sidebar.button("⬅️ Back to Home"): go_to('Home')
    st.title("Property Inquiry")
    col_in, col_out = st.columns([1, 1])
    with col_in:
        s = st.selectbox("State", sorted(state_city_map.keys()))
        c = st.selectbox("City", sorted(state_city_map[s]))
        sqm = st.number_input("Size (m²)", 10, 300, 75)
        rms = st.slider("Rooms", 1.0, 6.0, 2.5)
        k, b, li = st.checkbox("Kitchen"), st.checkbox("Balcony"), st.checkbox("Lift")
    with col_out:
        price = get_prediction(s, c, sqm, rms, k, b, li)
        st.header(f"Estimated Rent: €{price:,.2f}")

# ⚖️ EVALUATOR PAGE
elif st.session_state.page == 'Evaluator':
    if st.sidebar.button("⬅️ Back to Home"): go_to('Home')
    st.title("Market Deal Evaluator")
    s = st.selectbox("Property State", sorted(state_city_map.keys()))
    c = st.selectbox("Property City", sorted(state_city_map[s]))
    sqm = st.number_input("Property Size (m²)", 10, 300, 80)
    market_p = get_prediction(s, c, sqm, 3, 1, 1, 0)
    user_p = st.number_input("Enter Listing Price (€)", value=float(market_p))
    if user_p < market_p * 0.9: st.success("🔥 GREAT DEAL!")
    elif user_p > market_p * 1.1: st.error("⚠️ EXPENSIVE")
    else: st.warning("⚖️ Fair price")