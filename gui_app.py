import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import os
import json
from typing import List, Dict
import threading
from search import search_faiss_with_cost, clear_cache
from embedding import build_faiss_index
from main import load_book_with_pages

class TextEmbeddingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("書本文字搜尋小工具")
        self.root.geometry("1000x700")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.setup_ui()
        self.load_available_files()
        
    def setup_ui(self):
        """設定使用者介面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置網格權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # 標題
        title_label = ttk.Label(main_frame, text="文字嵌入搜尋系統", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 檔案選擇區域
        file_frame = ttk.LabelFrame(main_frame, text="檔案管理", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # 選擇索引檔案
        ttk.Label(file_frame, text="選擇索引檔案:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.index_var = tk.StringVar()
        self.index_combo = ttk.Combobox(file_frame, textvariable=self.index_var, 
                                       state="readonly", width=40)
        self.index_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # 重新整理按鈕
        refresh_btn = ttk.Button(file_frame, text="重新整理", 
                                command=self.load_available_files)
        refresh_btn.grid(row=0, column=2)
        
        # 建立新索引按鈕
        create_btn = ttk.Button(file_frame, text="建立新索引", 
                               command=self.create_new_index)
        create_btn.grid(row=1, column=0, pady=(10, 0), sticky=tk.W)
        
        # 搜尋區域
        search_frame = ttk.LabelFrame(main_frame, text="搜尋", padding="10")
        search_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(0, weight=1)
        
        # 搜尋輸入
        ttk.Label(search_frame, text="輸入查詢內容:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, font=('Arial', 11))
        search_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_entry.bind('<Return>', lambda e: self.perform_search())
        
        # 搜尋選項
        options_frame = ttk.Frame(search_frame)
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(options_frame, text="搜尋結果數量:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.top_k_var = tk.IntVar(value=3)
        top_k_spin = ttk.Spinbox(options_frame, from_=1, to=10, textvariable=self.top_k_var, width=5)
        top_k_spin.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # 搜尋按鈕
        search_btn = ttk.Button(options_frame, text="搜尋", 
                               command=self.perform_search, style='Accent.TButton')
        search_btn.grid(row=0, column=2, padx=(10, 0))
        
        # 清除快取按鈕
        clear_btn = ttk.Button(options_frame, text="清除快取", 
                              command=self.clear_cache)
        clear_btn.grid(row=0, column=3, padx=(10, 0))
        
        # 結果顯示區域
        result_frame = ttk.LabelFrame(main_frame, text="搜尋結果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 結果文字區域
        self.result_text = scrolledtext.ScrolledText(result_frame, 
                                                    font=('Arial', 10),
                                                    wrap=tk.WORD,
                                                    height=15)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 狀態列
        self.status_var = tk.StringVar(value="就緒")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def load_available_files(self):
        """
        載入可用的索引檔案，並更新下拉選單
        """
        try:
            embedding_dir = "embedding_files"
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)
                return
            
            index_files = []
            for file in os.listdir(embedding_dir):
                if file.endswith('.index'):
                    index_files.append(file)
            
            self.index_combo['values'] = index_files
            if index_files:
                self.index_combo.set(index_files[0])
            
            self.status_var.set(f"找到 {len(index_files)} 個索引檔案")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"載入檔案時發生錯誤: {str(e)}")
    
    def create_new_index(self):
        """
        建立新索引的對話框
        """
        dialog = CreateIndexDialog(self.root, self)
        self.root.wait_window(dialog.dialog)
        self.load_available_files()
    
    def perform_search(self):
        """
        執行搜尋
        """
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning("警告", "請輸入查詢內容")
            return
        
        index_file = self.index_var.get()
        if not index_file:
            messagebox.showwarning("警告", "請選擇索引檔案")
            return
        
        # 在新執行緒中執行搜尋，避免界面凍結
        self.status_var.set("搜尋中...")
        self.result_text.delete(1.0, tk.END)
        
        def search_thread():
            try:
                index_path = os.path.join("embedding_files", index_file)
                meta_path = index_path.replace('.index', '_meta.json')
                
                if not os.path.exists(meta_path):
                    self.root.after(0, lambda: messagebox.showerror("錯誤", f"找不到元數據檔案: {meta_path}"))
                    return
                
                # 執行搜尋
                results = search_faiss_with_cost(
                    query=query,
                    index_path=index_path,
                    meta_path=meta_path,
                    top_k=self.top_k_var.get()
                )
                
                # 更新結果顯示
                self.root.after(0, lambda: self.display_results(results))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("錯誤", f"搜尋時發生錯誤: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("搜尋完成"))
        
        threading.Thread(target=search_thread, daemon=True).start()
    
    def display_results(self, results):
        """
        顯示搜尋結果
        """
        self.result_text.delete(1.0, tk.END)
        
        # 顯示搜尋資訊
        info = f"查詢: {self.search_var.get()}\n"
        info += f"索引檔案: {self.index_var.get()}\n"
        info += f"搜尋結果數量: {len(results['results'])}\n"
        info += f"嵌入成本: ${results['embedding_cost_usd']:.8f} USD\n"
        info += f"總耗時: {results['total_time_sec']:.3f} 秒\n"
        info += "=" * 50 + "\n\n"
        
        self.result_text.insert(tk.END, info)
        
        # 顯示搜尋結果
        for i, result in enumerate(results['results'], 1):
            result_text = f"結果 {i}:\n"
            result_text += f"頁碼: {result.get('page', 'N/A')}\n"
            result_text += f"內容: {result.get('text', '')}\n"
            result_text += "-" * 50 + "\n\n"
            
            self.result_text.insert(tk.END, result_text)
    
    def clear_cache(self):
        """清除快取"""
        try:
            clear_cache()
            self.status_var.set("快取已清除")
            messagebox.showinfo("成功", "快取已清除")
        except Exception as e:
            messagebox.showerror("錯誤", f"清除快取時發生錯誤: {str(e)}")

class CreateIndexDialog:
    def __init__(self, parent, main_app):
        self.main_app = main_app
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("建立新索引")
        self.dialog.geometry("500x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 顯示置中
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        self.setup_dialog()
    
    def setup_dialog(self):
        """設定對話框介面"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 標題
        ttk.Label(main_frame, text="建立新的文字嵌入索引", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 20))
        
        # 選擇文字檔案
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="選擇文字檔案:").pack(anchor=tk.W)
        
        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(file_select_frame, textvariable=self.file_var, state="readonly")
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(file_select_frame, text="瀏覽", 
                  command=self.select_file).pack(side=tk.RIGHT)
        
        # 索引名稱
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(name_frame, text="索引名稱 (不含副檔名):").pack(anchor=tk.W)
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.name_var)
        name_entry.pack(fill=tk.X, pady=(5, 0))
        
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame, text="建立索引", 
                  command=self.create_index).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="取消", 
                  command=self.dialog.destroy).pack(side=tk.RIGHT)
    
    def select_file(self):
        """
        選擇文字檔案
        """
        file_path = filedialog.askopenfilename(
            title="選擇文字檔案",
            filetypes=[("文字檔案", "*.txt"), ("所有檔案", "*.*")],
            initialdir="txt_files"
        )
        if file_path:
            self.file_var.set(file_path)
            # 自動設定索引名稱
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            self.name_var.set(name_without_ext)
    
    def create_index(self):
        """
        建立索引
        """
        file_path = self.file_var.get()
        index_name = self.name_var.get().strip()
        
        if not file_path or not index_name:
            messagebox.showwarning("警告", "請選擇檔案並輸入索引名稱")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("錯誤", "選擇的檔案不存在")
            return
        
        try:
            # 在新執行緒中建立索引
            def create_thread():
                try:
                    self.dialog.after(0, lambda: self.dialog.title("建立索引中..."))
                    
                    # 載入並處理文字文件
                    chunks = load_book_with_pages(file_path)
                    
                    # 建立索引路徑
                    index_path = os.path.join("embedding_files", f"{index_name}.index")
                    meta_path = os.path.join("embedding_files", f"{index_name}_meta.json")
                    
                    # 建立 FAISS 索引
                    build_faiss_index(chunks, index_path, meta_path)
                    
                    self.dialog.after(0, lambda: messagebox.showinfo("成功", f"索引建立完成！\n檔案: {index_name}"))
                    self.dialog.after(0, lambda: self.dialog.destroy())
                    
                except Exception as e:
                    self.dialog.after(0, lambda: messagebox.showerror("錯誤", f"建立索引時發生錯誤: {str(e)}"))
                    self.dialog.after(0, lambda: self.dialog.title("建立新索引"))
            
            threading.Thread(target=create_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"建立索引時發生錯誤: {str(e)}")

def main():
    root = tk.Tk()
    app = TextEmbeddingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
