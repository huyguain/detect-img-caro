import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ChessboardAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện quân cờ trên bàn cờ")
        
        self.offset_ratio = 0.045
        self.cell_size_factor = (0.35, 0.45)
        self.thresholds = {'black': 100, 'white': 170}
        self.image = None
        self.processed_image = None
        
        self.create_widgets()

    def create_widgets(self):
        # Nút tải ảnh
        self.load_button = tk.Button(self.root, text="Tải ảnh", command=self.load_image)
        self.load_button.pack(pady=10)
        
        # Label để hiển thị ảnh
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # Nút xử lý ảnh
        self.process_button = tk.Button(self.root, text="Xử lý ảnh", command=self.process_image, state='disabled')
        self.process_button.pack(pady=10)
        
        # Text widget để hiển thị kết quả
        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
                return
            self.display_image(self.image)
            self.process_button.config(state='normal')

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((400, 400), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.photo)

    def detect_chessboard_region(self, image):
        height, width = image.shape[:2]
        top_offset = int(height * self.offset_ratio)
        bottom_offset = int(height * self.offset_ratio)
        left_offset = int(width * self.offset_ratio)
        right_offset = int(width * self.offset_ratio)

        cropped_h = height - top_offset - bottom_offset
        cropped_w = width - left_offset - right_offset
        avg_cell_size = int(round((cropped_h + cropped_w) / (2.0 * 8)))

        final_h = avg_cell_size * 8
        final_w = avg_cell_size * 8
        return image[top_offset:top_offset + final_h, left_offset:left_offset + final_w]

    def analyze_cell(self, cell):
        if cell.size == 0:
            return '-'

        gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        blurred_cell = cv2.medianBlur(gray_cell, 5)

        h, w = gray_cell.shape
        min_radius = int(min(h, w) * self.cell_size_factor[0])
        max_radius = int(min(h, w) * self.cell_size_factor[1])

        circles = cv2.HoughCircles(
            blurred_cell, cv2.HOUGH_GRADIENT, dp=1, minDist=min(h, w),
            param1=60, param2=12, minRadius=min_radius, maxRadius=max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))[0, 0]
            x, y, r = circles

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            masked_cell = cv2.bitwise_and(gray_cell, gray_cell, mask=mask)
            avg_gray = np.mean(masked_cell[mask == 255])

            if avg_gray < self.thresholds['black']:
                return 'B'
            elif avg_gray > self.thresholds['white']:
                return 'W'
        return '-'

    def analyze_board(self):
        if self.image is None:
            messagebox.showerror("Lỗi", "Vui lòng tải ảnh trước!")
            return None, None

        board_image = self.detect_chessboard_region(self.image)
        height, width = board_image.shape[:2]
        cell_height = height // 8
        cell_width = width // 8

        board = [['-' for _ in range(8)] for _ in range(8)]
        for row in range(8):
            for col in range(8):
                y1, y2 = row * cell_height, (row + 1) * cell_height
                x1, x2 = col * cell_width, (col + 1) * cell_width
                cell = board_image[y1:y2, x1:x2]
                board[row][col] = self.analyze_cell(cell)
                cv2.rectangle(board_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return board, board_image

    def process_image(self):
        board, processed_image = self.analyze_board()
        if board is None or processed_image is None:
            return

        self.processed_image = processed_image
        self.display_image(self.processed_image)

        self.result_text.delete(1.0, tk.END)
        # self.result_text.insert(tk.END, "Trạng thái bàn cờ:\n")
        self.result_text.insert(tk.END, "  A B C D E F G H\n")
        self.result_text.insert(tk.END, "  ---------------\n")
        for i, row in enumerate(board):
            self.result_text.insert(tk.END, f"{8-i} {' '.join(row)}\n")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessboardAnalyzer(root)
    app.run()