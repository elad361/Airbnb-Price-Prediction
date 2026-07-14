import os
import pickle

import numpy as np
import pandas as pd

for artifact in ('model.pkl', 'encoders.pkl'):
    if not os.path.exists(artifact):
        print(f"'{artifact}' not found. Run airbnb.py first.")
        raise SystemExit(1)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    enc = pickle.load(f)

le_neighbourhood    = enc['neighbourhood']
le_room             = enc['room_type']
neighbourhood_coords = enc['neighbourhood_coords']
bathrooms_median    = enc['bathrooms_median']

# ── INTERACTIVE UI ────────────────────────────────────────────────────────────
print("\n=== Airbnb Price Predictor (NYC) ===\n")

neighbourhoods = sorted(le_neighbourhood.classes_)
print("Neighbourhoods:")
for i, n in enumerate(neighbourhoods, 1):
    print(f"  {i:3}. {n}")
n_idx = int(input("\nEnter neighbourhood number: ")) - 1
neighbourhood_input = neighbourhoods[n_idx]
print(f"Selected: {neighbourhood_input}")

room_types = list(le_room.classes_)
print("\nRoom types:")
for i, r in enumerate(room_types, 1):
    print(f"  {i}. {r}")
r_idx = int(input("\nEnter room type number: ")) - 1
room_input = room_types[r_idx]
print(f"Selected: {room_input}")

accommodates = int(input("\nHow many guests? "))
bedrooms     = float(input("Number of bedrooms: "))
beds         = float(input("Number of beds: "))
bathrooms    = float(input(f"Number of bathrooms (press Enter for {bathrooms_median:.1f}): ") or bathrooms_median)
min_nights   = int(input("Minimum nights: "))

is_superhost = input("Is the host a superhost? (y/n): ").strip().lower() == 'y'

print("\nDo you have reviews yet?")
print("  1. Yes")
print("  2. No (new listing)")
has_reviews = input("Enter 1 or 2: ").strip()

if has_reviews == "2":
    review_score = 4.7
    num_reviews  = 0
    print(f"(Using default review score: {review_score})")
else:
    review_score = float(input("Review score (1-5, e.g. 4.5): "))
    num_reviews  = int(input("Number of reviews: "))

# Use the neighbourhood's median lat/lon
coords = neighbourhood_coords.get(neighbourhood_input, {'latitude': 40.73, 'longitude': -73.94})
lat = coords['latitude']
lon = coords['longitude']

# ── PREDICT ───────────────────────────────────────────────────────────────────
neighbourhood_enc = le_neighbourhood.transform([neighbourhood_input])[0]
room_enc          = le_room.transform([room_input])[0]

X_input = pd.DataFrame([{
    'neighbourhood_enc':     neighbourhood_enc,
    'room_type_enc':         room_enc,
    'accommodates':          accommodates,
    'bedrooms':              bedrooms,
    'beds':                  beds,
    'bathrooms':             bathrooms,
    'review_scores_rating':  review_score,
    'number_of_reviews':     num_reviews,
    'minimum_nights':        min_nights,
    'latitude':              lat,
    'longitude':             lon,
    'host_is_superhost':     int(is_superhost),
}])

predicted_price = np.expm1(model.predict(X_input)[0])

print(f"\n{'='*38}")
print(f"  Predicted price: ${predicted_price:,.0f} per night")
print(f"{'='*38}\n")
