from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['veterinary_db']

# Collections
users = db['users']
records = db['records']
health_records = db['health_records']  # New collection for health records

def add_sample_data():
    # Clear existing data
    records.delete_many({})
    
    # Sample data for animals
    sample_data = [
        {
            'animal_id': 'COW001',
            'animal_type': 'Cow',
            'breed': 'Holstein',
            'age': '4 years',
            'symptoms': 'Reduced milk production, Loss of appetite, High temperature 39.5°C',
            'diagnosis': 'Mastitis',
            'treatment': 'Antibiotics (Ceftiofur), Regular udder cleaning, Increased milking frequency',
            'date': datetime.now()
        },
        {
            'animal_id': 'COW002',
            'animal_type': 'Cow',
            'breed': 'Jersey',
            'age': '3 years',
            'symptoms': 'Lameness, Swollen hoof, Difficulty walking',
                'diagnosis': 'Foot rot',
            'treatment': 'Hoof trimming, Topical antibiotics, Clean dry bedding',
            'date': datetime.now()
        },
        {
            'animal_id': 'COW003',
            'animal_type': 'Cow',
            'breed': 'Gir',
            'age': '5 years',
            'symptoms': 'Diarrhea, Dehydration, Weakness',
            'diagnosis': 'Bacterial enteritis',
            'treatment': 'Oral electrolytes, Probiotics, Antibiotics course',
            'date': datetime.now()
        }
    ]
    
    # Insert sample data
    records.insert_many(sample_data)
    print("Sample animal medical records added successfully!")

if __name__ == '__main__':
    add_sample_data()
