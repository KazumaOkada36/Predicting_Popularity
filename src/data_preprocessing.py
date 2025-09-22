import requests
import json
import os
from typing import Dict, Optional
from textblob import TextBlob
import numpy as np

class RestaurantDataProcessor:
    """
    Real API integration for restaurant data collection
    """
    
    def __init__(self, google_api_key: Optional[str] = None):
        # API Keys - Only Google Places needed
        self.google_api_key = google_api_key or os.getenv('GOOGLE_PLACES_API_KEY')
        
        # API Endpoints
        self.places_base_url = "https://maps.googleapis.com/maps/api/place"
        
    def search_restaurant(self, restaurant_name: str, location: str) -> Dict:
        """
        Get real restaurant data from Google Places API only
        """
        if not self.google_api_key:
            print("Warning: No Google API key found. Using fake data.")
            return self._generate_fake_data(restaurant_name, location)
        
        try:
            # Get restaurant data from Google Places
            restaurant_data = self._search_google_places(restaurant_name, location)
            
            if 'error' in restaurant_data:
                return restaurant_data
            
            # Process the Google Places data
            processed_data = self._process_google_data(restaurant_data)
            
            return processed_data
            
        except Exception as e:
            print(f"API Error: {e}")
            return self._generate_fake_data(restaurant_name, location)
    
    def _search_google_places(self, restaurant_name: str, location: str) -> Dict:
        """
        Search Google Places API for restaurant data
        """
        try:
            # Step 1: Text search to find the restaurant
            search_url = f"{self.places_base_url}/textsearch/json"
            search_params = {
                'query': f"{restaurant_name} {location}",
                'type': 'restaurant',
                'key': self.google_api_key
            }
            
            print(f"Searching Google Places for: {restaurant_name} in {location}")
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            
            if search_response.status_code != 200:
                return {"error": f"Google Places API error: {search_response.status_code}"}
            
            if not search_data.get('results'):
                return {"error": "Restaurant not found in Google Places"}
            
            # Get the most relevant result
            place = search_data['results'][0]
            place_id = place['place_id']
            
            # Step 2: Get detailed information
            details_url = f"{self.places_base_url}/details/json"
            details_params = {
                'place_id': place_id,
                'fields': 'name,rating,user_ratings_total,price_level,types,geometry,formatted_address,reviews,photos,opening_hours',
                'key': self.google_api_key
            }
            
            details_response = requests.get(details_url, params=details_params)
            details_data = details_response.json()
            
            if 'result' not in details_data:
                return {"error": "Could not get restaurant details from Google Places"}
            
            result = details_data['result']
            
            # Process Google Places data
            processed_data = {
                'name': result.get('name', 'Unknown'),
                'address': result.get('formatted_address', 'Unknown'),
                'rating': result.get('rating', 0),
                'total_ratings': result.get('user_ratings_total', 0),
                'price_level': result.get('price_level', 2),
                'restaurant_types': result.get('types', []),
                'latitude': result.get('geometry', {}).get('location', {}).get('lat'),
                'longitude': result.get('geometry', {}).get('location', {}).get('lng'),
                'reviews': result.get('reviews', []),
                'photos': len(result.get('photos', [])),
                'is_open': self._parse_opening_hours(result.get('opening_hours', {})),
                'source': 'google_places'
            }
            
            return processed_data
            
        except Exception as e:
            print(f"Google Places API error: {e}")
            return {"error": f"Google Places API error: {str(e)}"}
    
    def _process_google_data(self, google_data: Dict) -> Dict:
        """
        Process Google Places data and create ML features
        """
        # Start with Google data
        processed = google_data.copy()
        
        # Use Google rating as the main rating
        processed['avg_rating'] = google_data.get('rating', 0)
        
        # Analyze review sentiment from Google reviews
        processed['sentiment_score'] = self._analyze_review_sentiment(google_data.get('reviews', []))
        
        # Estimate additional features based on Google data
        processed.update(self._estimate_additional_features(google_data))
        
        # Create ML features
        processed['features'] = self._extract_ml_features(processed)
        
        return processed
    
    def _estimate_additional_features(self, google_data: Dict) -> Dict:
        """
        Estimate features we don't get directly from Google Places
        These estimates make our ML model more realistic
        """
        rating = google_data.get('rating', 4.0)
        review_count = google_data.get('total_ratings', 50)
        price_level = google_data.get('price_level', 2)
        
        # Estimate foot traffic based on rating and reviews
        base_traffic = 50 + (rating - 1) * 40 + np.log10(review_count + 1) * 25
        
        # Adjust for price level (moderate prices = more traffic)
        price_multipliers = {1: 1.1, 2: 1.3, 3: 1.0, 4: 0.7}
        foot_traffic = int(base_traffic * price_multipliers.get(price_level, 1.0))
        
        # Estimate competition based on location type
        # (In real app, you'd use location data to determine this)
        types = google_data.get('restaurant_types', [])
        if any(t in ['tourist_attraction', 'point_of_interest'] for t in types):
            competition_density = np.random.normal(40, 10)  # Tourist area
        elif any(t in ['shopping_mall', 'establishment'] for t in types):
            competition_density = np.random.normal(50, 15)  # Commercial area
        else:
            competition_density = np.random.normal(25, 8)   # Regular area
        
        # Estimate social media presence based on popularity
        social_followers = int(review_count * np.random.uniform(2, 5))
        social_posts_per_week = max(1, int(rating - 2))
        
        return {
            'foot_traffic': max(20, foot_traffic),
            'competition_density': max(5, competition_density),
            'social_followers': social_followers,
            'social_posts_per_week': social_posts_per_week,
            'neighborhood_type': 'mixed',  # Default since we don't have exact data
            'cuisine_type': self._determine_cuisine_type(types)
        }
    
    def _determine_cuisine_type(self, google_types: list) -> str:
        """
        Determine cuisine type from Google Places types
        """
        # Map Google types to cuisine categories
        cuisine_mapping = {
            'meal_takeaway': 'fast_food',
            'pizza': 'pizza',
            'cafe': 'cafe',
            'bar': 'bar',
            'bakery': 'bakery'
        }
        
        for gtype in google_types:
            if gtype in cuisine_mapping:
                return cuisine_mapping[gtype]
        
        return 'american'  # Default
    
    def _analyze_review_sentiment(self, reviews: list) -> float:
        """
        Analyze sentiment of Google reviews using TextBlob
        """
        if not reviews:
            return 0.5  # Neutral sentiment
        
        sentiments = []
        for review in reviews[:10]:  # Analyze up to 10 reviews
            text = review.get('text', '')
            if text:
                blob = TextBlob(text)
                # TextBlob sentiment polarity: -1 (negative) to 1 (positive)
                # Convert to 0-1 scale
                sentiment = (blob.sentiment.polarity + 1) / 2
                sentiments.append(sentiment)
        
        return np.mean(sentiments) if sentiments else 0.5
    
    def _extract_ml_features(self, restaurant_data: Dict) -> Dict:
        """
        Extract features for ML model from real API data
        """
        features = {}
        
        # Basic features
        features['avg_rating'] = restaurant_data.get('avg_rating', 0)
        features['total_reviews'] = restaurant_data.get('total_ratings', 0)
        features['price_level'] = restaurant_data.get('price_level', 2)
        
        # Location features
        features['has_coordinates'] = 1 if restaurant_data.get('latitude') else 0
        
        # Engagement features
        features['review_density'] = (
            features['total_reviews'] / (features['avg_rating'] + 0.1) 
            if features['avg_rating'] > 0 else 0
        )
        
        # Content features
        features['has_photos'] = 1 if restaurant_data.get('photos', 0) > 0 else 0
        features['sentiment_score'] = restaurant_data.get('sentiment_score', 0.5)
        
        # Business features
        features['is_open'] = 1 if restaurant_data.get('is_open', True) else 0
        
        # Type popularity (simplified)
        popular_types = ['restaurant', 'meal_takeaway', 'food', 'establishment']
        features['type_popularity_score'] = sum(
            1 for t in restaurant_data.get('restaurant_types', []) 
            if t in popular_types
        )
        
        return features
    
    def _parse_opening_hours(self, opening_hours: Dict) -> bool:
        """
        Parse opening hours to determine if restaurant is currently open
        """
        return opening_hours.get('open_now', True)
    
    def _generate_fake_data(self, restaurant_name: str, location: str) -> Dict:
        """
        Generate realistic fake data when APIs aren't available
        """
        import random
        
        fake_data = {
            'name': restaurant_name,
            'address': f"123 Main St, {location}",
            'rating': round(random.uniform(3.5, 4.8), 1),
            'total_ratings': random.randint(20, 500),
            'price_level': random.randint(1, 4),
            'restaurant_types': ['restaurant', 'food', 'establishment'],
            'latitude': round(random.uniform(40.0, 41.0), 6),
            'longitude': round(random.uniform(-74.5, -73.5), 6),
            'reviews': [],
            'photos': random.randint(5, 20),
            'is_open': True,
            'sentiment_score': round(random.uniform(0.4, 0.8), 2),
            'source': 'fake_data'
        }
        
        # Add features
        fake_data['features'] = self._extract_ml_features(fake_data)
        
        return fake_data


# ==========================================
# USAGE EXAMPLE
# ==========================================

if __name__ == "__main__":
    # Set your API key as environment variable or pass directly
    # os.environ['GOOGLE_PLACES_API_KEY'] = 'your_api_key_here'
    
    processor = RestaurantDataProcessor()
    
    # Test with real restaurant
    result = processor.search_restaurant("Joe's Pizza", "New York, NY")
    
    print("Restaurant Data:")
    print(json.dumps(result, indent=2))
    
    if 'features' in result:
        print("\nML Features:")
        for key, value in result['features'].items():
            print(f"  {key}: {value}")