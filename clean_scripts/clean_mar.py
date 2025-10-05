#!/usr/bin/env python3
import os
import re
import gc
import argparse
import time
from pathlib import Path

class MarathiTextCleaner:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = 100 * 1024 * 1024  # 100MB chunks
        self.seen_hashes = set()
        self.write_buffer = []
        self.buffer_size = 3000
        
        # Devanagari patterns
        self.devanagari_chars = re.compile(
            r'[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]'  # Devanagari Unicode ranges
        )
        
        self.non_devanagari = re.compile(
            r'[^\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF\s\.\,\!\?\;\:\-\(\)\"\'\।]'
        )
        
        # English detection patterns
        self.english_word = re.compile(r'[a-zA-Z]{3,}')  # Words with 3+ English letters
        self.multi_space = re.compile(r'\s{2,}')
        self.multi_punct = re.compile(r'([\.\,\!\?\;]){3,}')
        
        # Statistics
        self.stats = {
            'total_lines': 0,
            'lines_kept': 0,
            'total_tokens': 0,
            'english_removed': 0,
            'duplicates_removed': 0,
            'start_time': time.time()
        }

    def contains_english(self, text):
        """Check if text contains English words"""
        return bool(self.english_word.search(text))

    def remove_english_words(self, text):
        """Remove English words from text"""
        return self.english_word.sub('', text)

    def is_marathi_text(self, text, threshold=0.6):
        """Check if text is primarily Marathi Devanagari"""
        if not text or len(text.strip()) < 10:
            return False
        
        devanagari_count = len(self.devanagari_chars.findall(text))
        total_chars = len(text.strip())
        
        return (devanagari_count / total_chars) >= threshold

    def clean_marathi_text(self, text):
        """Clean Marathi text with English removal"""
        # Remove English words first
        text = self.remove_english_words(text)
        
        # Remove unwanted characters
        text = self.non_devanagari.sub('', text)
        
        # Normalize spaces
        text = self.multi_space.sub(' ', text)
        
        # Reduce excessive punctuation
        text = self.multi_punct.sub(r'\1\1', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text

    def count_marathi_tokens(self, text):
        """Count tokens in Marathi text (words + punctuation)"""
        # Split by whitespace and count non-empty elements
        return len([word for word in text.split() if word.strip()])

    def calculate_simple_hash(self, text):
        """Lightweight deduplication hash"""
        sample = text[:200].lower().replace(' ', '').replace('\n', '')
        return hash(sample)

    def is_good_marathi_line(self, text, min_words=3):
        """Check if line meets quality criteria"""
        if len(text) < 15 or len(text) > 2000:
            return False
        
        words = len(text.split())
        if words < min_words:
            return False
        
        return bool(self.devanagari_chars.search(text))

    def flush_buffer(self, outfile):
        """Write buffer to file"""
        if self.write_buffer:
            outfile.writelines(line + '\n' for line in self.write_buffer)
            outfile.flush()
            self.write_buffer.clear()

    def print_progress(self, bytes_processed, file_size, chunk_kept, chunk_lines):
        """Print detailed progress information"""
        progress_percent = (bytes_processed / file_size) * 100
        elapsed_time = time.time() - self.stats['start_time']
        
        # Calculate ETA
        if progress_percent > 0:
            total_time_estimate = (elapsed_time / progress_percent) * 100
            eta_seconds = total_time_estimate - elapsed_time
            eta_str = f"{eta_seconds//3600:.0f}h {(eta_seconds%3600)//60:.0f}m"
        else:
            eta_str = "calculating..."
        
        # Calculate processing speed
        mb_processed = bytes_processed / (1024 * 1024)
        speed_mb_per_sec = mb_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Current chunk retention
        retention = (chunk_kept / chunk_lines * 100) if chunk_lines > 0 else 0
        
        # Token estimates
        avg_tokens_per_line = (self.stats['total_tokens'] / self.stats['lines_kept']) if self.stats['lines_kept'] > 0 else 0
        estimated_total_tokens = self.stats['lines_kept'] * avg_tokens_per_line
        
        print(f"\n Progress: {progress_percent:.1f}% ({mb_processed:.0f}MB/{file_size/(1024*1024):.0f}MB)")
        print(f" ETA: {eta_str} | Speed: {speed_mb_per_sec:.1f} MB/s")
        print(f" Lines: {self.stats['total_lines']:,} total, {self.stats['lines_kept']:,} kept")
        print(f" Tokens: ~{estimated_total_tokens/1000000:.1f}M estimated")
        print(f" Removed: {self.stats['english_removed']:,} English, {self.stats['duplicates_removed']:,} duplicates")
        print(f" Retention: {retention:.1f}% (chunk) | {(self.stats['lines_kept']/self.stats['total_lines']*100):.1f}% (overall)")
        print("-" * 80)

    def process_file(self):
        file_size = os.path.getsize(self.input_file)
        print(f"  Marathi dataset size: {file_size / (1024**3):.2f} GB")
        print(" Starting Marathi text cleaning...")
        print("=" * 80)
        
        bytes_processed = 0
        chunk_count = 0
        
        with open(self.input_file, 'rb') as infile, \
             open(self.output_file, 'w', encoding='utf-8', buffering=16384) as outfile:
            
            while True:
                chunk = infile.read(self.chunk_size)
                if not chunk:
                    break
                
                # Handle chunk boundaries
                if len(chunk) == self.chunk_size:
                    last_newline = chunk.rfind(b'\n')
                    if last_newline > 0:
                        infile.seek(infile.tell() - (len(chunk) - last_newline))
                        chunk = chunk[:last_newline]
                
                chunk_lines = 0
                chunk_kept = 0
                
                for line in chunk.split(b'\n'):
                    chunk_lines += 1
                    self.stats['total_lines'] += 1
                    
                    try:
                        text = line.decode('utf-8', errors='ignore').strip()
                        if not text:
                            continue
                        
                        # Remove English content
                        if self.contains_english(text):
                            self.stats['english_removed'] += 1
                            continue
                        
                        cleaned = self.clean_marathi_text(text)
                        
                        if not self.is_good_marathi_line(cleaned) or not self.is_marathi_text(cleaned):
                            continue
                        
                        # Deduplication
                        text_hash = self.calculate_simple_hash(cleaned)
                        if text_hash in self.seen_hashes:
                            self.stats['duplicates_removed'] += 1
                            continue
                        
                        self.seen_hashes.add(text_hash)
                        self.write_buffer.append(cleaned)
                        chunk_kept += 1
                        self.stats['lines_kept'] += 1
                        
                        # Count tokens for this line
                        tokens = self.count_marathi_tokens(cleaned)
                        self.stats['total_tokens'] += tokens
                        
                        if len(self.write_buffer) >= self.buffer_size:
                            self.flush_buffer(outfile)
                            
                    except Exception:
                        continue
                
                bytes_processed += len(chunk)
                chunk_count += 1
                
                # Print progress every chunk
                self.print_progress(bytes_processed, file_size, chunk_kept, chunk_lines)
                
                # Memory management
                if chunk_count % 5 == 0:
                    self.flush_buffer(outfile)
                    if len(self.seen_hashes) > 80000:
                        self.seen_hashes.clear()
                    gc.collect()
            
            # Final flush
            self.flush_buffer(outfile)
        
        # Final statistics
        self.print_final_stats()

    def print_final_stats(self):
        """Print comprehensive final statistics"""
        elapsed_time = time.time() - self.stats['start_time']
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        avg_tokens_per_line = (self.stats['total_tokens'] / self.stats['lines_kept']) if self.stats['lines_kept'] > 0 else 0
        
        print("=" * 80)
        print("✅ MARATHI CLEANING COMPLETE!")
        print("=" * 80)
        print(f" Time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f" Total lines processed: {self.stats['total_lines']:,}")
        print(f" Lines kept: {self.stats['lines_kept']:,}")
        print(f" Estimated total tokens: {self.stats['total_tokens']:,} (~{self.stats['total_tokens']/1000000:.1f}M)")
        print(f" Average tokens per line: {avg_tokens_per_line:.1f}")
        print(f" Retention rate: {(self.stats['lines_kept']/self.stats['total_lines']*100):.1f}%")
        print(f" English content removed: {self.stats['english_removed']:,} lines")
        print(f" Duplicates removed: {self.stats['duplicates_removed']:,} lines")
        print(f" Output file: {self.output_file}")
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Clean large Marathi text dataset with progress tracking')
    parser.add_argument('input_file', help='Path to input Marathi text file')
    parser.add_argument('output_file', help='Path to output cleaned file')
    parser.add_argument('--chunk-size', type=int, default=100, 
                       help='Chunk size in MB (default: 100)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f" Error: Input file '{args.input_file}' does not exist")
        return
    
    chunk_size_bytes = args.chunk_size * 1024 * 1024
    
    cleaner = MarathiTextCleaner(args.input_file, args.output_file)
    cleaner.chunk_size = chunk_size_bytes
    cleaner.process_file()

if __name__ == "__main__":
    main()
