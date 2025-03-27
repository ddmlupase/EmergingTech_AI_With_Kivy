import cv2
import numpy as np
import json
import requests
from tensorflow.keras.models import load_model
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle

# Load ML model and class indices
model = load_model('fruit_recognition_model.h5')
fruit_classes = {
    0: "Banana",
    1: "Eggplant",
    2: "Okra",
    3: "Onion",
    4: "Tomato"
}

def get_nutrition(fruit_name):
    api_key = "AruAFSTx9fcRdxMaHf5I9p696DotbfO8W1v2HWYp"
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"query": fruit_name, "api_key": api_key, "pageSize": 1}
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        foods = data.get("foods", [])
        if foods:
            nutrients = foods[0].get("foodNutrients", [])
            nutrition_info = {nutrient["nutrientName"]: nutrient["value"] for nutrient in nutrients[:5]}  # Now 5 items
            return nutrition_info
    return {"error": "No data found."}

def get_recipes(fruit_name):
    api_key = "5627cb51d3a8493ab16e1564684ec273"  # Replace with your Spoonacular API key
    url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "query": fruit_name,
        "apiKey": api_key,
        "number": 5  # Only 5 recipes
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        recipes = response.json().get("results", [])
        return [recipe["title"] for recipe in recipes] if recipes else ["No recipes found."]
    return ["No recipes found."]  # Handle errors properly

class CameraApp(App):
    def build(self):
        layout = BoxLayout(orientation="vertical")  # Main layout
        
        # Set background color to #303A50 (RGB 48, 58, 80)
        with layout.canvas.before:
            Color(0.188, 0.227, 0.314, 1)  # Converted RGBA
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
        layout.bind(size=self.update_rect, pos=self.update_rect)

        self.camera = Camera(resolution=(640, 480), play=False)
        self.btn = Button(text="Start", size_hint=(1, 0.1))
        self.btn.bind(on_press=self.toggle_camera)

        self.label = Label(text="Press Start to Detect", size_hint=(1, 0.1))

        # Create a horizontal layout for Nutrition & Recipes
        info_layout = BoxLayout(orientation="horizontal", size_hint=(1, 0.3))  

        self.nutrition_label = Label(text="Nutrition Info: ", size_hint=(0.5, 1))  # 50% width
        self.recipes_label = Label(text="Recipes: ", size_hint=(0.5, 1))  # 50% width

        info_layout.add_widget(self.nutrition_label)
        info_layout.add_widget(self.recipes_label)

        layout.add_widget(self.camera)
        layout.add_widget(self.label)
        layout.add_widget(info_layout)  # Added horizontal layout here
        layout.add_widget(self.btn)

        return layout

    def toggle_camera(self, instance):
        if not self.camera.play:
            self.camera.play = True
            self.btn.text = 'Stop'
            Clock.schedule_interval(self.detect_fruit, 1)  # Run detection every 1 sec
        else:
            self.camera.play = False
            self.btn.text = 'Start'
            Clock.unschedule(self.detect_fruit)

    def detect_fruit(self, dt):
        frame = self.camera.texture  # Get frame from Kivy Camera
        
        if frame:
            frame = frame.pixels  # Convert to image array
            frame = np.frombuffer(frame, dtype=np.uint8).reshape((480, 640, 4))  # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Process image for model
            resized = cv2.resize(frame, (100, 100))
            normalized = resized / 255.0
            input_tensor = np.expand_dims(normalized, axis=0)

            predictions = model.predict(input_tensor, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            fruit_name = fruit_classes[class_idx]
            label = f"{fruit_name} ({confidence:.2f})"
            
            # Update labels
            self.label.text = label
            
            # Get and display nutrition info (now 5 nutrients)
            nutrition = get_nutrition(fruit_name)
            nutrition_items = list(nutrition.items())[:5]  # Now showing 5 items
            nutrition_text = "\n".join([f"{key}: {value}" for key, value in nutrition_items]) if nutrition_items else "Not Found"
            self.nutrition_label.text = f"Nutrition Info:\n{nutrition_text}"
            
            # Get and display recipes (only 5 recipes)
            recipes = get_recipes(fruit_name)
            recipes_text = "\n".join(recipes[:5])  # Get first 5 recipes
            self.recipes_label.text = f"Recipes:\n{recipes_text}"

    def update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

if __name__ == "__main__":
    CameraApp().run()
