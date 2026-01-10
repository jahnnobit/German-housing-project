# DE Real Estate Pro: AI Rental Predictor

A full-stack Data Science application that predicts rental prices across Germany using Machine Learning. This project processes a massive dataset to provide users with real-time market insights and deal evaluations.



## Live Demo
**Check out the live app here:** [https://german-housing-project-ayqygnohbugidfpzuttxrx.streamlit.app/](https://german-housing-project-ayqygnohbugidfpzuttxrx.streamlit.app/)

---

## Project Overview
This application utilizes a **Linear Regression model** trained on over **270,000 property listings** from across Germany. It allows users to estimate fair market rent based on specific apartment features and compare pricing trends between different federal states and cities.

### Key Features
* **Rent Prediction:** Input features like square meters, number of rooms, and amenities (balcony, lift, kitchen) to get an AI-calculated price.
* **City Comparison:** Side-by-side analysis of rental markets in different German cities.
* **Deal Evaluator:** Determine if a specific listing is overpriced or a bargain compared to the statistical average.
* **Dynamic Visuals:** Interactive charts showing average rent distribution across the country.

---

## Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Linear Regression)
* **Data Handling:** Pandas, NumPy
* **Large File Storage:** Git LFS (Used to manage 271MB dataset)
* **Web Framework:** Streamlit
* **Visualization:** Plotly

---

## Architecture & Challenges
One of the primary challenges of this project was managing the **271MB dataset**, which exceeds GitHub's standard upload limit. 
1.  **Git LFS:** Implemented Large File Storage to track and version the data without bloat.
2.  **Cloud Training:** Configured a custom pipeline so the model trains automatically upon the first deployment on the Streamlit Cloud server.
3.  **UI/UX:** Injected custom CSS into Streamlit to create a professional, user-friendly real estate dashboard.

---

##  Project Structure
* `app.py`: The main Streamlit application code.
* `immo_data.csv`: The full dataset (managed via LFS).
* `requirements.txt`: Necessary libraries for deployment.
* `website home page.jpg`: Custom UI background assets.

---

##  Contact
**Jahnnobit** - [GitHub Profile](https://github.com/jahnnobit)
