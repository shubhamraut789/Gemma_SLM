import os
import re
import gc
import unicodedata
import hashlib
import random
import shutil

class LowStorageTextCleaner:
    def __init__(self, input_file, output_file, chunk_size=200*1024*1024):
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.seen_hashes = set()
        self.buffer_size = 5000
        self.write_buffer = []

        # Regex patterns
        self.multi_space = re.compile(r'\s{2,}')
        self.html_tag = re.compile(r'<[^>]+>')
        self.zero_width = re.compile(r'[\u200B-\u200D\uFEFF]')

    def is_good_line(self, line: str) -> bool:
        """Basic filter to drop noisy/short/long lines"""
        if len(line) < 5 or len(line) > 1000:
            return False
        words = line.count(' ') + 1
        if words < 2:
            return False
        return any(c.isalpha() for c in line)

    def fast_clean(self, line: str) -> str:
        """Normalize, strip HTML, zero-width chars, collapse spaces"""
        line = unicodedata.normalize("NFC", line)
        line = self.html_tag.sub('', line)
        line = self.zero_width.sub('', line)
        line = self.multi_space.sub(' ', line)
        return line.strip()

    def flush_buffer(self, outfile):
        if self.write_buffer:
            outfile.writelines(line + '\n' for line in self.write_buffer)
            outfile.flush()
            self.write_buffer.clear()

    def line_hash(self, line: str) -> str:
        """Hash line for deduplication"""
        return hashlib.md5(line.encode('utf-8')).hexdigest()

    def clean_data(self):
        file_size = os.path.getsize(self.input_file)
        print(f"Input file size: {file_size / (1024**3):.2f} GB")

        total_processed, total_kept = 0, 0
        sample_tokens, sample_lines = 0, 0
        bytes_processed, chunk_num = 0, 0

        with open(self.input_file, 'rb') as infile, \
             open(self.output_file, 'w', encoding='utf-8', buffering=32768) as outfile:

            while True:
                chunk = infile.read(self.chunk_size)
                if not chunk:
                    break

                # Handle boundary cut
                if len(chunk) == self.chunk_size:
                    last_newline = chunk.rfind(b'\n')
                    if last_newline > 0:
                        infile.seek(infile.tell() - (len(chunk) - last_newline))
                        chunk = chunk[:last_newline]

                chunk_processed, chunk_kept = 0, 0

                for line in chunk.split(b'\n'):
                    chunk_processed += 1
                    try:
                        text = line.decode('utf-8', errors='ignore').strip()
                        if not text or not self.is_good_line(text):
                            continue

                        cleaned = self.fast_clean(text)
                        if not cleaned:
                            continue

                        h = self.line_hash(cleaned)
                        if h not in self.seen_hashes:
                            self.seen_hashes.add(h)
                            self.write_buffer.append(cleaned)
                            chunk_kept += 1

                            # Sampling for token estimate
                            if chunk_kept % 100 == 0:
                                sample_tokens += len(cleaned.split())
                                sample_lines += 1

                            if len(self.write_buffer) >= self.buffer_size:
                                self.flush_buffer(outfile)
                    except:
                        continue

                total_processed += chunk_processed
                total_kept += chunk_kept
                chunk_num += 1
                bytes_processed += len(chunk)

                progress = (bytes_processed / file_size) * 100
                overall_retention = (total_kept / total_processed * 100) if total_processed else 0

                if sample_lines > 0:
                    avg_tokens = sample_tokens / sample_lines
                    est_tokens = total_kept * avg_tokens
                    token_info = f"Est.Tokens: {est_tokens/1_000_000:.0f}M"
                else:
                    token_info = "Est.Tokens: calculating..."

                print(f"Batch {chunk_num} | Progress {progress:.1f}% | "
                      f"Kept {chunk_kept}/{chunk_processed} | "
                      f"Total {total_kept:,} | Ret {overall_retention:.1f}% | {token_info}")

                if chunk_num % 20 == 0:
                    self.flush_buffer(outfile)
                    if len(self.seen_hashes) > 1_000_000:
                        print("Clearing dedup cache...")
                        self.seen_hashes.clear()
                    gc.collect()

            self.flush_buffer(outfile)

        # Save stats
        stats_path = os.path.join(os.path.dirname(self.output_file), "end_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("=== Cleaning Stats ===\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Output file: {self.output_file}\n")
            f.write(f"File size: {file_size / (1024**3):.2f} GB\n")
            f.write(f"Total processed: {total_processed}\n")
            f.write(f"Total kept: {total_kept}\n")
            if sample_lines > 0:
                avg_tokens = sample_tokens / sample_lines
                est_tokens = total_kept * avg_tokens
                f.write(f"Estimated tokens: {est_tokens:,.0f}\n")
                f.write(f"Average tokens/line: {avg_tokens:.1f}\n")
            f.write(f"Overall retention: {(total_kept/total_processed)*100:.2f}%\n\n")

        print(f"\nCleaning complete. Final lines kept: {total_kept:,}")


def shuffle_and_split_large(input_file, out_dir, chunk_size=200*1024*1024,
                            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    chunk_files = []
    chunk_id = 0

    print("Step 1: Splitting into shuffled chunks...")
    with open(input_file, "r", encoding="utf-8") as f:
        buffer, buffer_size = [], 0
        for line in f:
            buffer.append(line)
            buffer_size += len(line.encode("utf-8"))
            if buffer_size >= chunk_size:
                random.shuffle(buffer)
                temp_path = os.path.join(temp_dir, f"chunk_{chunk_id}.txt")
                with open(temp_path, "w", encoding="utf-8") as out:
                    out.writelines(buffer)
                chunk_files.append(temp_path)
                chunk_id += 1
                buffer, buffer_size = [], 0
        if buffer:
            random.shuffle(buffer)
            temp_path = os.path.join(temp_dir, f"chunk_{chunk_id}.txt")
            with open(temp_path, "w", encoding="utf-8") as out:
                out.writelines(buffer)
            chunk_files.append(temp_path)

    print(f"Created {len(chunk_files)} shuffled chunks.")

    print("Step 2: Merging shuffled chunks...")
    random.shuffle(chunk_files)
    shuffled_path = os.path.join(out_dir, "shuffled.txt")
    with open(shuffled_path, "w", encoding="utf-8") as out:
        for cf in chunk_files:
            with open(cf, "r", encoding="utf-8") as tempf:
                for line in tempf:
                    out.write(line)

    print("Step 3: Splitting into train/val/test...")
    with open(shuffled_path, "r", encoding="utf-8") as f:
        lines = sum(1 for _ in f)

    n_train = int(lines * train_ratio)
    n_val = int(lines * val_ratio)

    train_file = open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8")
    val_file = open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8")
    test_file = open(os.path.join(out_dir, "test.txt"), "w", encoding="utf-8")

    with open(shuffled_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < n_train:
                train_file.write(line)
            elif i < n_train + n_val:
                val_file.write(line)
            else:
                test_file.write(line)

    train_file.close()
    val_file.close()
    test_file.close()

    # Append stats
    stats_path = os.path.join(out_dir, "end_stats.txt")
    with open(stats_path, "a", encoding="utf-8") as f:
        f.write("=== Splitting Stats ===\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Shuffled file: {shuffled_path}\n")
        f.write(f"Total lines: {lines}\n")
        f.write(f"Train: {n_train}\n")
        f.write(f"Val: {n_val}\n")
        f.write(f"Test: {lines - n_train - n_val}\n")
        f.write(f"Ratios used: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}\n")
        f.write(f"Chunk size: {chunk_size/(1024*1024):.1f} MB\n")
        f.write(f"Temp chunks created: {len(chunk_files)}\n")
        f.write(f"Output directory: {out_dir}\n\n")

    # Cleanup
    print("Cleaning up temporary files...")
    try:
        os.remove(shuffled_path)
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: cleanup failed - {e}")

    print(f"Split complete: Train={n_train}, Val={n_val}, Test={lines - n_train - n_val}")
    print(f"Stats saved in {stats_path}")


if __name__ == "__main__":
    input_file = "merged_english_corpus.txt" 
    cleaned_file = "cleaned_english.txt"

    cleaner = LowStorageTextCleaner(input_file, cleaned_file)
    cleaner.clean_data()

    shuffle_and_split_large(cleaned_file, out_dir="finaleng_splits", chunk_size=200*1024*1024)
