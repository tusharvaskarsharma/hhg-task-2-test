# RAG Accuracy Results

This report evaluates the accuracy of the production HHG RAG pipeline using the `miracl/miracl` dataset.

## Language: HI
- **Evaluated Queries:** 100

### Metrics
| Pipeline | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **HNSW** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **RRF** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Grounding Evaluation
### Examples of Missed Relevant Documents
**Query:** कांग्रेस दल का नेता कौन है ?
- Gold IDs: ['31181#4', '659567#0', '6394#10']
- Retrieved Top 5: ['6f45116eeb481bbb4099cc39', 'c2a646b54332f1bc3f96343f', 'b3b98f417ae8d436401aeb2e', '536ecc0c77ea49296441cea9', 'cc9d87a7768180e55f7d2ff4']

**Query:** पाकिस्तान में कोनसा धर्म सबसे बढ़ा है?
- Gold IDs: ['495305#10']
- Retrieved Top 5: ['dc222424d9a96689cfc7c384', 'c2a646b54332f1bc3f96343f', '547c5424b1fc6a18f0ffc05e', '536ecc0c77ea49296441cea9', '9dcc262360104298cc4a9c0f']

**Query:** गूगल की खोज किसने की थी ?
- Gold IDs: ['220491#0']
- Retrieved Top 5: ['0cf5f0d3f9fa23874c2fb13a', 'c2a646b54332f1bc3f96343f', '2e18d6a6c7b2d8846d136df6', '536ecc0c77ea49296441cea9', '0ef5f0176fec5a44a7bd9fc1']

---

## Language: BN
- **Evaluated Queries:** 100

### Metrics
| Pipeline | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **HNSW** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **RRF** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Grounding Evaluation
### Examples of Missed Relevant Documents
**Query:** ইংরেজ আন্তর্জাতিক ক্রিকেট তারকা জর্জ গিবসন ম্যাকাউলি কি একজন ডানহাতি ব্যাটসম্যান ছিলেন ?
- Gold IDs: ['717942#0']
- Retrieved Top 5: ['0ff811d4259d172382ad20a3', 'f243f722948c0f502fc14801', '861fa42e189d7f7d0ee59db5', '074bcf363546a4a74bfb3842', '68d98c1d2c34af99d3808618']

**Query:** জাতিসংঘ শান্তিরক্ষা মিশন প্রথম কোথায় শুরু হয় ?
- Gold IDs: ['625049#0']
- Retrieved Top 5: ['3a3b73091572a6771dee8010', 'f243f722948c0f502fc14801', '63fb8283a3b618fc09d3fb74', '074bcf363546a4a74bfb3842', '60d0524c4dd7a905a608b77e']

**Query:** ডাব্লিউডাব্লিউই ক্রীঢ়া বিনোদন টেলিভিশন প্রোগ্রামটি কবে প্রথম চালু হয় ?
- Gold IDs: ['452067#0', '378894#2', '452331#0']
- Retrieved Top 5: ['5abd4bddb20449a2e082bdf2', 'f243f722948c0f502fc14801', '3a3b73091572a6771dee8010', '074bcf363546a4a74bfb3842', '63fb8283a3b618fc09d3fb74']

---

