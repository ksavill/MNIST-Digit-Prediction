import tkinter as tk
from PIL import Image, ImageDraw
import torch
from model import SimpleCNN
from utils import preprocess_image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint with 28x28 Grid")

        # Set a fixed geometry for the window so it doesn't change later
        self.root.geometry("800x600")

        self.grid_count = 28
        self.cell_size = 10
        self.canvas_size = self.grid_count * self.cell_size

        # -----------
        # Brush Size Slider
        # -----------
        self.brush_size_var = tk.IntVar(value=16)  # default value = 16
        self.color = "black"

        # Create a Scale widget to let the user adjust brush size
        self.brush_slider = tk.Scale(
            root,
            from_=1,        # minimum brush size
            to=50,          # maximum brush size
            orient=tk.HORIZONTAL,
            label="Brush Size",
            variable=self.brush_size_var,
            command=self.update_brush_size
        )
        self.brush_slider.pack()

        # Canvas for drawing
        self.canvas = tk.Canvas(root, bg="white",
                                width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Draw grid lines so the user sees 28 distinct cells
        for i in range(self.grid_count + 1):
            pos = i * self.cell_size
            self.canvas.create_line(pos, 0, pos, self.canvas_size, fill="lightgray")
            self.canvas.create_line(0, pos, self.canvas_size, pos, fill="lightgray")

        # Create a PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse event for drawing
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons (Reset & Predict)
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_canvas)
        self.reset_button.pack()

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack()

        # Label to show predictions
        self.prediction_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.prediction_label.pack()

        # Activation plot area
        self.activation_frame = tk.Frame(root, width=500, height=300)
        self.activation_frame.pack(fill=tk.BOTH, expand=True)

        default_fig = Figure(figsize=(5, 4))
        default_ax = default_fig.add_subplot(111)
        default_ax.text(0.5, 0.5, "Activation plots will appear here",
                        ha="center", va="center")
        default_ax.axis('off')
        self.activation_canvas = FigureCanvasTkAgg(default_fig, master=self.activation_frame)
        self.activation_canvas.draw()
        self.activation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # For debouncing predictions while drawing
        self.predict_after_id = None

        # Load the trained model
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load('model_best.pth', map_location="cpu"))
        self.model.eval()

        # Set default brush size
        self.brush_size = self.brush_size_var.get()

    # -----------
    # Brush size update method
    # -----------
    def update_brush_size(self, value):
        self.brush_size = int(value)

    def paint(self, event):
        x1 = event.x - self.brush_size
        y1 = event.y - self.brush_size
        x2 = event.x + self.brush_size
        y2 = event.y + self.brush_size

        # Draw on both Tkinter canvas and the PIL image
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.color, outline=self.color)
        self.draw.ellipse([x1, y1, x2, y2], fill=0, outline=0)

        # Debounce predictions (only run after short delay)
        if self.predict_after_id is not None:
            self.root.after_cancel(self.predict_after_id)
        self.predict_after_id = self.root.after(200, self.predict)

    def reset_canvas(self):
        self.canvas.delete("all")
        # Redraw grid lines
        for i in range(self.grid_count + 1):
            pos = i * self.cell_size
            self.canvas.create_line(pos, 0, pos, self.canvas_size, fill="lightgray")
            self.canvas.create_line(0, pos, self.canvas_size, pos, fill="lightgray")

        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="")

        # Cancel any scheduled prediction
        if self.predict_after_id is not None:
            self.root.after_cancel(self.predict_after_id)
            self.predict_after_id = None

        # Reset the activation plot
        if self.activation_canvas is not None:
            self.activation_canvas.get_tk_widget().destroy()
        default_fig = Figure(figsize=(5, 4))
        default_ax = default_fig.add_subplot(111)
        default_ax.text(0.5, 0.5, "Activation plots will appear here",
                        ha="center", va="center")
        default_ax.axis('off')
        self.activation_canvas = FigureCanvasTkAgg(default_fig, master=self.activation_frame)
        self.activation_canvas.draw()
        self.activation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def predict(self):
        # Preprocess the image (including centering, resizing, etc.)
        img_tensor = preprocess_image(self.image)
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(dim=1).item()

        self.prediction_label.config(text=f"Predicted Digit: {pred}")

        # Display feature maps or activations
        fig = self.get_activations_figure(self.model, img_tensor)
        if self.activation_canvas is not None:
            self.activation_canvas.get_tk_widget().destroy()
        self.activation_canvas = FigureCanvasTkAgg(fig, master=self.activation_frame)
        self.activation_canvas.draw()
        self.activation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def get_activations_figure(self, model, input_tensor):
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks on conv1 and conv2
        hook1 = model.conv1.register_forward_hook(get_activation('conv1'))
        hook2 = model.conv2.register_forward_hook(get_activation('conv2'))

        # Run model to collect activations
        with torch.no_grad():
            _ = model(input_tensor)

        # Remove hooks
        hook1.remove()
        hook2.remove()

        # Create a figure with 2 subplots for conv1 and conv2
        fig = Figure(figsize=(5, 4))
        axes = fig.subplots(2, 1)

        # Plot first 8 feature maps from conv1
        act_conv1 = activations['conv1'].squeeze().cpu().numpy()
        for i in range(min(act_conv1.shape[0], 8)):
            axes[0].plot(act_conv1[i])
        axes[0].set_title('Conv1 Activations')

        # Plot first 8 feature maps from conv2
        act_conv2 = activations['conv2'].squeeze().cpu().numpy()
        for i in range(min(act_conv2.shape[0], 8)):
            axes[1].plot(act_conv2[i])
        axes[1].set_title('Conv2 Activations')

        fig.tight_layout()
        return fig

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()