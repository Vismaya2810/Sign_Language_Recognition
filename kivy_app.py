from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.label = Label(text="Press the button to import MediaPipe and OpenCV")
        self.add_widget(self.label)
        btn = Button(text="Test Imports")
        btn.bind(on_press=self.test_imports)
        self.add_widget(btn)

    def test_imports(self, instance):
        try:
            import mediapipe as mp
            import cv2
            self.label.text = "MediaPipe and OpenCV imported successfully!"
        except Exception as e:
            self.label.text = f"Error importing mediapipe or cv2: {e}"

class TestApp(App):
    def build(self):
        return MainLayout()

if __name__ == '__main__':
    TestApp().run() 