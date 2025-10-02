import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import matplotlib.pyplot as plt

# ------------- Developer Info ----------------
DEV_PHOTO = "sonia.png"
DEV_NAME = "Tanvir Akter Sonia"
DEV_ID = "0812220105101068"
# ---------------------------------------------

# ------------ Utility Functions --------------
def pil_to_np(img): return np.array(img)
def np_to_pil(arr): return Image.fromarray(np.uint8(arr))
def ensure_rgb(img): return img.convert("RGB") if img.mode != "RGB" else img
# ---------------------------------------------

# ------------ Image Operations ---------------
def negative(arr): return 255 - arr

def grayscale(arr):
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = (0.299*r+0.587*g+0.114*b).astype(np.uint8)
    return np.stack([gray]*3, axis=-1)

def threshold(arr, limit=128):
    gray = (0.299*arr[:, :, 0]+0.587*arr[:, :, 1]+0.114*arr[:, :, 2])
    mask = gray >= limit
    out = np.zeros_like(arr)
    out[mask] = [255, 255, 255]
    return out

def resize_img(arr, w, h): return np.array(Image.fromarray(arr).resize((w, h)))

def convolve(arr, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
    out = np.zeros_like(arr, dtype=np.float32)
    kf = np.flipud(np.fliplr(kernel))
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            region = padded[y:y+kh, x:x+kw]
            for c in range(3):
                out[y, x, c] = np.sum(region[:, :, c]*kf)
    return np.clip(out, 0, 255).astype(np.uint8)

def sharpen(arr, intensity=1.0):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])*intensity
    return convolve(arr, kernel)

def smooth(arr, ksize=3):
    kernel = np.ones((ksize, ksize))/(ksize*ksize)
    return convolve(arr, kernel)

def log_transform(arr, c=1):
    arr = arr.astype(np.float32)
    out = c * np.log1p(arr)
    out = 255 * out / np.max(out)
    return np.clip(out, 0, 255).astype(np.uint8)

def gamma_transform(arr, gamma=1.0):
    arr = arr.astype(np.float32) / 255.0
    out = np.power(arr, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def edge_detection(arr):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gray = grayscale(arr)[:, :, 0]
    grad_x = convolve(np.stack([gray]*3, axis=-1), gx)[:, :, 0]
    grad_y = convolve(np.stack([gray]*3, axis=-1), gy)[:, :, 0]
    mag = np.sqrt(grad_x**2 + grad_y**2)
    mag = (mag / np.max(mag)) * 255
    return np.stack([mag.astype(np.uint8)]*3, axis=-1)

def show_histogram(arr):
    gray = grayscale(arr)[:, :, 0]
    plt.figure("Histogram")
    plt.hist(gray.ravel(), bins=256, range=(0, 255), color='blue')
    plt.title("Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
# ---------------------------------------------

# ------------ GUI Class ----------------------
class ToolkitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("✨ GUI-based Image Processing Toolkit ✨")
        self.geometry("1550x850")
        self.config(bg="#a3c3de")   # Light theme

        self.orig_pil = None
        self.orig_np = None
        self.proc_pil = None
        self.proc_np = None
        self.saved_path = None  

        self.build_gui()

    def build_gui(self):
        # ---------- Title ----------
        title_frame = tk.Frame(self, bg="#6abdca", bd=4, relief="groove")
        title_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        tk.Label(title_frame, text="✨ GUI-based Image Processing Toolkit ✨ ",
                 font=("Helvetica", 20, "bold"), fg="white", bg="#0D404C", padx=10, pady=8).pack()

        # ---------- Top Row: Dev Info + Buttons ----------
        top_row = tk.Frame(self, bg="#DCF0F0")
        top_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Developer Info
        dev_frame = tk.Frame(top_row, bg="#f0f8ff")
        dev_frame.pack(side=tk.LEFT, padx=20)
        canvas = tk.Canvas(dev_frame, width=150, height=150, bg="white", bd=2, relief="solid")
        canvas.pack(pady=5)
        try:
            if os.path.exists(DEV_PHOTO):
                img = Image.open(DEV_PHOTO).resize((150, 150))
                self.dev_img = ImageTk.PhotoImage(img)
                canvas.create_image(75, 75, image=self.dev_img)
            else:
                canvas.create_text(75, 75, text="No Photo")
        except:
            canvas.create_text(75, 75, text="Error")
        tk.Label(dev_frame, text=DEV_NAME, font=("Arial", 12, "bold"), bg="#f0f8ff").pack()
        tk.Label(dev_frame, text=DEV_ID, font=("Arial", 11), bg="#f0f8ff").pack()

        # Buttons Area
        btn_frame = tk.Frame(top_row, bg="#f0f8ff")
        btn_frame.pack(side=tk.LEFT, padx=30)

        btn_colors = ["#ff6666", "#66b3ff", "#99cc66", "#ff9933", "#cc99ff", "#66cccc",
                      "#ffcc66", "#66cc99", "#ff6699", "#3399ff", "#ff9966"]

        ops = [
            ("Select Image", self.load_img),
            ("Negative", self.do_negative),
            ("Grayscale", self.do_gray),
            ("Thresholding", self.do_thresh),
            ("Sharpening", self.do_sharp),
            ("Smoothing", self.do_smooth),
            ("Histogram", self.do_histogram),
            ("Log Transform", self.do_log),
            ("Gamma Transform", self.do_gamma),
            ("Edge Detection", self.do_edge)
        ]

        # 1st row
        row1 = tk.Frame(btn_frame, bg="#c5ddf3"); row1.pack(pady=3)
        for i, (name, cmd) in enumerate(ops[:6]):
            tk.Button(row1, text=name, command=cmd, bg=btn_colors[i], fg="white",
                      font=("Arial", 11, "bold"), width=16, pady=6, relief="raised", bd=2).pack(side=tk.LEFT, padx=4)

        # 2nd row
        row2 = tk.Frame(btn_frame, bg="#9ab6ce"); row2.pack(pady=3)
        for i, (name, cmd) in enumerate(ops[6:], start=6):
            tk.Button(row2, text=name, command=cmd, bg=btn_colors[i], fg="white",
                      font=("Arial", 11, "bold"), width=16, pady=6, relief="raised", bd=2).pack(side=tk.LEFT, padx=4)

        # --- Resize Option Row ---
        resize_row = tk.Frame(btn_frame, bg="#f0f8ff"); resize_row.pack(pady=8)
        tk.Label(resize_row, text="Width:", bg="#f0f8ff", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.width_entry = tk.Entry(resize_row, width=6)
        self.width_entry.insert(0, "300")
        self.width_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(resize_row, text="Height:", bg="#f0f8ff", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.height_entry = tk.Entry(resize_row, width=6)
        self.height_entry.insert(0, "300")
        self.height_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(resize_row, text="Apply Resize", command=self.do_resize,
                  bg="#0099cc", fg="white", font=("Arial", 11, "bold"), width=15).pack(side=tk.LEFT, padx=8)

        # Save / Show row
        row3 = tk.Frame(btn_frame, bg="#f0f8ff"); row3.pack(pady=10)
        tk.Button(row3, text="💾 Save Enhanced Image", command=self.save_img,
                  bg="#009966", fg="white", font=("Arial", 12, "bold"),
                  width=20, pady=8, relief="raised", bd=3).pack(side=tk.LEFT, padx=8)
        tk.Button(row3, text="📂 Show Saved Image", command=self.show_saved_img,
                  bg="#cc3333", fg="white", font=("Arial", 12, "bold"),
                  width=20, pady=8, relief="raised", bd=3).pack(side=tk.LEFT, padx=8)

        # ---------- Middle: Image Panels ----------
        middle = tk.Frame(self, bg="#f0f8ff")
        middle.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.left_canvas = self.add_image_panel(middle, "Original Image", 0)
        self.right_canvas = self.add_image_panel(middle, "Enhanced Image", 1)
        self.save_canvas = self.add_image_panel(middle, "Saved Image", 2)

    def add_image_panel(self, parent, title, col):
        frame = tk.LabelFrame(parent, text=title, bg="#ffffff", font=("Arial", 12, "bold"), fg="#333366", bd=3, relief="groove")
        frame.grid(row=0, column=col, padx=15, pady=10)
        canvas = tk.Canvas(frame, width=420, height=360, bg="#f9f9f9")
        canvas.pack()
        return canvas

    # -------- Display Helpers --------
    def show_original(self):
        if self.orig_pil:
            img = self.orig_pil.resize((420, 360))
            self.tk_orig = ImageTk.PhotoImage(img)
            self.left_canvas.create_image(0, 0, anchor="nw", image=self.tk_orig)

    def show_processed(self):
        if self.proc_pil:
            img = self.proc_pil.resize((420, 360))
            self.tk_proc = ImageTk.PhotoImage(img)
            self.right_canvas.create_image(0, 0, anchor="nw", image=self.tk_proc)

    def show_saved_img(self):
        if self.saved_path and os.path.exists(self.saved_path):
            saved_pil = Image.open(self.saved_path).resize((420, 360))
            self.tk_saved = ImageTk.PhotoImage(saved_pil)
            self.save_canvas.create_image(0, 0, anchor="nw", image=self.tk_saved)

    # -------- Actions --------
    def load_img(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.orig_pil = ensure_rgb(Image.open(path))
        self.orig_np = pil_to_np(self.orig_pil)
        self.proc_np = self.orig_np.copy()
        self.proc_pil = np_to_pil(self.proc_np)
        self.show_original()
        self.show_processed()

    def save_img(self):
        if self.proc_pil:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                self.proc_pil.save(path)
                self.saved_path = path
                messagebox.showinfo("Saved", f"Image saved at {path}")

    def apply_and_show(self, func):
        self.proc_np = func(self.proc_np)
        self.proc_pil = np_to_pil(self.proc_np)
        self.show_processed()

    # ---- Image Operation Buttons ----
    def do_negative(self): self.apply_and_show(negative)
    def do_gray(self): self.apply_and_show(grayscale)
    def do_thresh(self): self.apply_and_show(lambda arr: threshold(arr, 128))
    
    def do_resize(self):
        try:
            w = int(self.width_entry.get())
            h = int(self.height_entry.get())
            self.apply_and_show(lambda arr: resize_img(arr, w, h))
        except:
            messagebox.showerror("Error", "Invalid width/height values")

    def do_sharp(self): self.apply_and_show(lambda arr: sharpen(arr, 1.0))
    def do_smooth(self): self.apply_and_show(lambda arr: smooth(arr, 3))
    def do_histogram(self): show_histogram(self.proc_np)
    def do_log(self): self.apply_and_show(log_transform)
    def do_gamma(self): self.apply_and_show(lambda arr: gamma_transform(arr, 1.0))
    def do_edge(self): self.apply_and_show(edge_detection)


# ----------------- Run -----------------------
if __name__ == "__main__":
    app = ToolkitApp()
    app.mainloop()
