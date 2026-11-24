# Eksperimen Kuantitatif: LLM Code Review dengan Oracle AI Vector Search 23ai

**Tugas UAS:** LLM - Code Review - Mandiri  
**Mahasiswa:** Muhammad Hafizh Ramadhan  
**NIM:** 12409011030031  
**Program Studi:** Teknik Informatika  
**Akses repositori:** Permintaan berbagi akses untuk `rizalbroer@uinjkt.ac.id`. Undangan akses harus dikirim melalui pengaturan repositori GitHub oleh pemilik repositori; dokumentasi ini tidak mengirim undangan.

## Ringkasan dan metode ATM

Proyek ini menerapkan metode **Amati-Tiru-Modifikasi (ATM)** pada proses code review berbantuan LLM. Tahap *Amati* mengacu pada Collante et al. (2025), *The Impact of Large Language Models (LLMs) on Code Review Process* ([arXiv:2508.11034](https://arxiv.org/abs/2508.11034)), yang membahas perubahan durasi pada fase-fase siklus review. Tahap *Tiru* mengoperasionalkan perbandingan waktu melalui data PR simulasi, regresi linear berganda, dan uji Mann–Whitney U. Tahap *Modifikasi* menambahkan konsep Oracle AI Vector Search 23ai: embedding ONNX, indeks HNSW, pencarian PR serupa, dan augmentasi konteks RAG.

**Research gap dan novelty yang diuji:** eksperimen ini mengusulkan penghubung eksplisit antara pencarian konteks PR historis berbasis vector search Oracle 23ai dan metrik kuantitatif per fase code review. Angka dan hasil di repositori ini berasal dari **simulasi terkontrol**, bukan reproduksi data atau temuan empiris paper. Simulasi mengatur penurunan rata-rata fase review sekitar 66,78%, waktu tunggu sebelum perubahan 86,36%, dan total waktu hingga merge 48,32%; fase perubahan kode hanya sekitar 3,77%. Klaim novelty adalah motivasi/hipotesis rancangan yang perlu diuji dengan data nyata, bukan klaim bahwa integrasi Oracle telah dijalankan atau tervalidasi secara empiris.

### Variabel dataset

| Variabel | Definisi |
|---|---|
| `pr_id` | ID PR sintetis. |
| `is_rag_assisted` | Indikator perlakuan: `1` RAG-assisted, `0` non-assisted. |
| `project_age_month` | Usia proyek dalam bulan. |
| `no_commits` | Jumlah commit pada PR. |
| `PR_size` | Ukuran perubahan sintetis (baris perubahan). |
| `no_changed_files` | Jumlah file yang berubah. |
| `at_submission_h` | Durasi dari commit pertama sampai PR dibuka, dalam jam. |
| `at_review_h` | Durasi sampai respons review pertama, dalam jam. |
| `at_waiting_change_h` | Waktu diskusi dan menunggu revisi, dalam jam. |
| `at_change_h` | Waktu revisi kode hingga diterima, dalam jam. |
| `at_waiting_after_h` | Waktu setelah acceptance sampai merge, dalam jam. |
| `time_to_merge_h` | Total durasi semua fase di atas, dalam jam. |
| `vector_cosine_sim` | Similarity cosine contoh hasil retrieval; kosong untuk kontrol non-RAG. |
| `manhattan_match_dist` | Jarak Manhattan ke PR terdekat dari kelompok perlakuan berlawanan setelah fitur kontrol dinormalisasi. |
| `task_*` | Kategori tugas code review per fase: Enhancement, Bug Fix, Documentation, Implementation, Testing, atau Other. |

### Diagram 1: Arsitektur Pipeline (Oracle AI Vector Search + RAG Code Review)

```mermaid
flowchart LR
    subgraph Input["1. Input & Data Preprocessing"]
        A[("GitHub PR Dataset<br/>(n = 300 PRs)")] --> B["Feature Extraction & Rescaling<br/>(Min-Max [0,1] & Manhattan Dist)"]
        A --> C["Text & Code Diff Extraction<br/>(PR Title + Modified Code)"]
    end

    subgraph Oracle23ai["2. Oracle AI Vector Search 23ai (OA Module)"]
        C --> D["ONNX Embedding Model<br/>VECTOR_EMBEDDING()"]
        D --> E[("Oracle Table<br/>VECTOR(384, FLOAT32)")]
        E --> F["HNSW Neighbor Graph Index<br/>(Cosine Distance, Acc 95%)"]
    end

    subgraph RAG["3. Augmented Generation & Review"]
        F -- "Top-K Similar PRs" --> G["Prompt Augmentation<br/>(System + Historical Context)"]
        G --> H["LLM Code Reviewer<br/>(Enhancement / Bug Fix / Docs)"]
    end

    subgraph Eval["4. Analisis Kuantitatif (Paper Wajib)"]
        B & H --> I["Multiple Linear Regression<br/>log1p(time_to_merge)"]
        B & H --> J["Mann-Whitney U Test<br/>(At Review, Waiting, Change)"]
    end

    style Oracle23ai fill:#eff6ff,stroke:#1e3a8a,stroke-width:2px
    style Eval fill:#fefce8,stroke:#d97706,stroke-width:2px
```

### Diagram 2: Alur Siklus Hidup PR & Titik Intervensi RAG

```mermaid
flowchart TD
    S1["<b>Fase 1: At Submission</b><br/>First Commit &rarr; Open PR<br/><i>(Tidak ada intervensi LLM)</i>"]
    S2["<b>Fase 2: At Review (First Response)</b><br/>Open PR &rarr; Review Pertama<br/><b>Efisiensi: -66,78% (2,83j &rarr; 0,94j)</b><br/><i>Oracle Vector Search cari PR serupa + LLM review awal</i>"]
    S3["<b>Fase 3: At Waiting (Before Change)</b><br/>Diskusi Multi-Turn & Tunggu Revisi<br/><b>Efisiensi Terbesar: -86,36% (21,99j &rarr; 3,00j)</b><br/><i>Rekomendasi RAG mempercepat pemahaman developer</i>"]
    S4["<b>Fase 4: At Change (Modification)</b><br/>Revisi Kode &rarr; Acceptance<br/><b>Selisih: -3,77% (1,06j &rarr; 1,02j | Tidak Signifikan)</b><br/><i>Developer tetap compile & unit test lokal</i>"]
    S5["<b>Fase 5: Post-Acceptance & Merge</b><br/>Acceptance &rarr; Merged<br/><b>Total Time to Merge turun -48,32%</b>"]

    S1 --> S2 --> S3 --> S4 --> S5

    style S2 fill:#dbeafe,stroke:#1e3a8a,stroke-width:2px
    style S3 fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## Struktur proyek

```text
.
├── data/
│   └── pr_dataset.csv                 # Dibuat oleh generator
├── outputs/                           # Laporan statistik dan grafik
├── prompts/
│   └── code_review_prompts.md
├── sql/
│   └── oracle_23ai_vector_rag.sql
├── src/
│   ├── generate_dataset.py
│   └── run_experiment.py
└── requirements.txt
```

## Menjalankan eksperimen

Gunakan Python 3.10+ dari direktori repositori:

```bash
python -m pip install -r requirements.txt
python src/generate_dataset.py
python src/run_experiment.py
```

Generator membuat 300 baris simulasi seimbang (150 per kelompok) dengan seed 42. Analisis menghasilkan `outputs/regression_summary.txt`, `outputs/mann_whitney_results.csv`, `outputs/task_category_counts.csv`, `outputs/manhattan_matching_summary.txt`, serta tiga grafik PNG 300 DPI. Skrip SQL adalah contoh implementasi untuk Oracle Database 23ai dan memerlukan tabel, model embedding ONNX `ALL_MINILM_L6_V2`, serta akses database yang sesuai; SQL tidak dijalankan oleh eksperimen Python lokal.

## Referensi

- Collante et al. (2025). [The Impact of Large Language Models (LLMs) on Code Review Process](https://arxiv.org/abs/2508.11034). arXiv:2508.11034.
- Oracle Academy. *Introduction to Vector Search with Oracle Database 23ai* (materi pembelajaran Oracle Academy).
