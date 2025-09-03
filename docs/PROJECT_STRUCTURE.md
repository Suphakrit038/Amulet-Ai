# Amulet-AI Project Structure

This document describes the structure of the Amulet-AI project after reorganization.

## Directory Structure

```
Project Structure

├── 📁 ai_models
│   ├── 📁 configs
│   │   ├── ⚙️ config_advanced.json
│   │   ├── 📄 requirements_advanced.txt
│   │   └── 📄 requirements_minimal.txt
│   ├── 📁 core
│   │   ├── 🧠 amulet_model.h5
│   │   ├── 🧠 amulet_model.tflite
│   │   └── ⚙️ labels.json
│   ├── 📁 dataset_split
│   │   ├── 📁 test
│   │   │   ├── 📁 somdej-fatherguay
│   │   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1.jpg
│   │   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1.jpg
│   │   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1(bw).jpg
│   │   │   │   └── 🖼️ somdejsrewaree-fatherguay-b2(bw).jpg
│   │   │   ├── 📁 พระพุทธเจ้าในวิหาร
│   │   │   │   ├── 🖼️ Phraphuthjao in viharn1front.png
│   │   │   │   └── 🖼️ Phraphuthjao in viharn3front.png
│   │   │   ├── 📁 พระสมเด็จฐานสิงห์
│   │   │   │   ├── 🖼️ somdejthansing 1front.png
│   │   │   │   └── 🖼️ somdejthansing 2back.png
│   │   │   ├── 📁 พระสมเด็จประทานพร พุทธกวัก
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg12-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg3-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg3-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg4-back.png
│   │   │   │   └── 🖼️ Prasomdej-pudtagueg8-front.png
│   │   │   ├── 📁 พระสมเด็จหลังรูปเหมือน
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 1back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 2back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 3front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 5front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 7front.png
│   │   │   │   └── 🖼️ prasomejhaunlouphmuan 8front.png
│   │   │   ├── 📁 พระสรรค์
│   │   │   │   └── 🖼️ prasan 1back.png
│   │   │   ├── 📁 พระสิวลี
│   │   │   │   ├── 🖼️ prasrewaree 10front.png
│   │   │   │   ├── 🖼️ prasrewaree 3front.png
│   │   │   │   ├── 🖼️ prasrewaree 4front.png
│   │   │   │   └── 🖼️ prasrewaree 6back.png
│   │   │   ├── 📁 สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 10front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 1back.png
│   │   │   │   └── 🖼️ somdejpimprougpo9bai 8back.png
│   │   │   ├── 📁 สมเด็จแหวกม่าน
│   │   │   │   ├── 🖼️ somdejHæwkman16front.png
│   │   │   │   ├── 🖼️ somdejHæwkman1front.png
│   │   │   │   ├── 🖼️ somdejHæwkman20front.png
│   │   │   │   ├── 🖼️ somdejHæwkman7front.png
│   │   │   │   └── 🖼️ somdejHæwkman8front.png
│   │   │   └── 📁 ออกวัดหนองอีดุก
│   │   │       ├── 🖼️ watnongEduk10back.png
│   │   │       ├── 🖼️ watnongEduk2front.png
│   │   │       ├── 🖼️ watnongEduk8back.png
│   │   │       ├── 🖼️ watnongEduk8front.png
│   │   │       └── 🖼️ watnongEduk9front.png
│   │   ├── 📁 train
│   │   │   ├── 📁 somdej-fatherguay
│   │   │   │   ├── 🖼️ somdej-beginning-fatherguay-b1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-beginning-fatherguay-b1.jpg
│   │   │   │   ├── 🖼️ somdej-beginning-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-beginning-fatherguay-f1.jpg
│   │   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-b1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-b1.jpg
│   │   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1.jpg
│   │   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-b1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-b1.jpg
│   │   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-f1.jpg
│   │   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-b1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-b1.jpg
│   │   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-f1.jpg
│   │   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-b1(bw).jpg
│   │   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-b1.jpg
│   │   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1.jpg
│   │   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1.jpg
│   │   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2(bw).jpg
│   │   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2.jpg
│   │   │   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-b1(bw).jpg
│   │   │   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-b1.jpg
│   │   │   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-f1.jpg
│   │   │   │   ├── 🖼️ somdejsrewaree-fatherguay-b2(bw).jpg
│   │   │   │   ├── 🖼️ somdejsrewaree-fatherguay-b2.jpg
│   │   │   │   ├── 🖼️ somdejsrewaree-fatherguay-f1(bw).jpg
│   │   │   │   └── 🖼️ somdejsrewaree-fatherguay-f1.jpg
│   │   │   ├── 📁 พระพุทธเจ้าในวิหาร
│   │   │   │   ├── 🖼️ Phraphuthjao in viharn1front.png
│   │   │   │   ├── 🖼️ Phraphuthjao in viharn2front.png
│   │   │   │   ├── 🖼️ Phraphuthjao in viharn3front.png
│   │   │   │   ├── 🖼️ Phraphuthjao in viharn4front.png
│   │   │   │   ├── 🖼️ Phraphuthjao in viharn5front.png
│   │   │   │   ├── 🖼️ Phraphuthjao in viharn6back.png
│   │   │   │   └── 🖼️ Phraphuthjao in viharn7back.png
│   │   │   ├── 📁 พระสมเด็จฐานสิงห์
│   │   │   │   ├── 🖼️ somdejthansing 1back.png
│   │   │   │   ├── 🖼️ somdejthansing 1front.png
│   │   │   │   ├── 🖼️ somdejthansing 2back.png
│   │   │   │   └── 🖼️ somdejthansing 2front.png
│   │   │   ├── 📁 พระสมเด็จประทานพร พุทธกวัก
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg1-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg1-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg10-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg10-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg11-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg11-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg12-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg13-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg13-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg2-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg2-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg3-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg3-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg4-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg4-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg5-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg5-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg6-back.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg6-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg7-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg8-front.png
│   │   │   │   └── 🖼️ Prasomdej-pudtagueg9-front.png
│   │   │   ├── 📁 พระสมเด็จหลังรูปเหมือน
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 10front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 11back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 11front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 12back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 12front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 13back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 13front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 1back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 1front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 2back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 2front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 3back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 3front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 4back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 4front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 5back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 5front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 6back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 6front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 7back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 7front.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 8back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 9back.png
│   │   │   │   └── 🖼️ prasomejhaunlouphmuan 9front.png
│   │   │   ├── 📁 พระสรรค์
│   │   │   │   ├── 🖼️ prasan 1back.png
│   │   │   │   └── 🖼️ prasan 1front.png
│   │   │   ├── 📁 พระสิวลี
│   │   │   │   ├── 🖼️ prasrewaree 10front.png
│   │   │   │   ├── 🖼️ prasrewaree 1back.png
│   │   │   │   ├── 🖼️ prasrewaree 1front.png
│   │   │   │   ├── 🖼️ prasrewaree 2back.png
│   │   │   │   ├── 🖼️ prasrewaree 2front.png
│   │   │   │   ├── 🖼️ prasrewaree 3back.png
│   │   │   │   ├── 🖼️ prasrewaree 3front.png
│   │   │   │   ├── 🖼️ prasrewaree 4back.png
│   │   │   │   ├── 🖼️ prasrewaree 4front.png
│   │   │   │   ├── 🖼️ prasrewaree 5back.png
│   │   │   │   ├── 🖼️ prasrewaree 5front.png
│   │   │   │   ├── 🖼️ prasrewaree 6back.png
│   │   │   │   ├── 🖼️ prasrewaree 6front.png
│   │   │   │   ├── 🖼️ prasrewaree 7front.png
│   │   │   │   ├── 🖼️ prasrewaree 8front.png
│   │   │   │   └── 🖼️ prasrewaree 9front.png
│   │   │   ├── 📁 สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 10back.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 10front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 1front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 2back.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 2front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 3back.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 3front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 4back.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 4front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 5back.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 5front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 6back.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 6front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 7back.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 7front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 8front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 9back.png
│   │   │   │   └── 🖼️ somdejpimprougpo9bai 9front.png
│   │   │   ├── 📁 สมเด็จแหวกม่าน
│   │   │   │   ├── 🖼️ somdejHæwkman10front.png
│   │   │   │   ├── 🖼️ somdejHæwkman11front.png
│   │   │   │   ├── 🖼️ somdejHæwkman12front.png
│   │   │   │   ├── 🖼️ somdejHæwkman13front.png
│   │   │   │   ├── 🖼️ somdejHæwkman14front.png
│   │   │   │   ├── 🖼️ somdejHæwkman15front.png
│   │   │   │   ├── 🖼️ somdejHæwkman17front.png
│   │   │   │   ├── 🖼️ somdejHæwkman18front.png
│   │   │   │   ├── 🖼️ somdejHæwkman19front.png
│   │   │   │   ├── 🖼️ somdejHæwkman20front.png
│   │   │   │   ├── 🖼️ somdejHæwkman21front.png
│   │   │   │   ├── 🖼️ somdejHæwkman2front.png
│   │   │   │   ├── 🖼️ somdejHæwkman3front.png
│   │   │   │   ├── 🖼️ somdejHæwkman4front.png
│   │   │   │   ├── 🖼️ somdejHæwkman5front.png
│   │   │   │   ├── 🖼️ somdejHæwkman6front.png
│   │   │   │   ├── 🖼️ somdejHæwkman7front.png
│   │   │   │   ├── 🖼️ somdejHæwkman8front.png
│   │   │   │   └── 🖼️ somdejHæwkman9front.png
│   │   │   └── 📁 ออกวัดหนองอีดุก
│   │   │       ├── 🖼️ watnongEduk10back.png
│   │   │       ├── 🖼️ watnongEduk10front.png
│   │   │       ├── 🖼️ watnongEduk1back.png
│   │   │       ├── 🖼️ watnongEduk1front.png
│   │   │       ├── 🖼️ watnongEduk2back.png
│   │   │       ├── 🖼️ watnongEduk2front.png
│   │   │       ├── 🖼️ watnongEduk3back.png
│   │   │       ├── 🖼️ watnongEduk3front.png
│   │   │       ├── 🖼️ watnongEduk4back.png
│   │   │       ├── 🖼️ watnongEduk4front.png
│   │   │       ├── 🖼️ watnongEduk5back.png
│   │   │       ├── 🖼️ watnongEduk5front.png
│   │   │       ├── 🖼️ watnongEduk6front.png
│   │   │       ├── 🖼️ watnongEduk7back.png
│   │   │       ├── 🖼️ watnongEduk7front.png
│   │   │       ├── 🖼️ watnongEduk8back.png
│   │   │       ├── 🖼️ watnongEduk8front.png
│   │   │       ├── 🖼️ watnongEduk9back.png
│   │   │       └── 🖼️ watnongEduk9front.png
│   │   ├── 📁 validation
│   │   │   ├── 📁 somdej-fatherguay
│   │   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1(bw).jpg
│   │   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1.jpg
│   │   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2(bw).jpg
│   │   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2.jpg
│   │   │   │   └── 🖼️ somdejsrewaree-fatherguay-f1.jpg
│   │   │   ├── 📁 พระพุทธเจ้าในวิหาร
│   │   │   │   └── 🖼️ Phraphuthjao in viharn4front.png
│   │   │   ├── 📁 พระสมเด็จประทานพร พุทธกวัก
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg4-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg5-front.png
│   │   │   │   ├── 🖼️ Prasomdej-pudtagueg7-front.png
│   │   │   │   └── 🖼️ Prasomdej-pudtagueg9-front.png
│   │   │   ├── 📁 พระสมเด็จหลังรูปเหมือน
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 10back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 6back.png
│   │   │   │   ├── 🖼️ prasomejhaunlouphmuan 8front.png
│   │   │   │   └── 🖼️ prasomejhaunlouphmuan 9back.png
│   │   │   ├── 📁 พระสิวลี
│   │   │   │   ├── 🖼️ prasrewaree 5front.png
│   │   │   │   ├── 🖼️ prasrewaree 6front.png
│   │   │   │   └── 🖼️ prasrewaree 8front.png
│   │   │   ├── 📁 สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 1back.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 3front.png
│   │   │   │   ├── 🖼️ somdejpimprougpo9bai 7front.png
│   │   │   │   └── 🖼️ somdejpimprougpo9bai 9front.png
│   │   │   ├── 📁 สมเด็จแหวกม่าน
│   │   │   │   ├── 🖼️ somdejHæwkman10front.png
│   │   │   │   ├── 🖼️ somdejHæwkman16front.png
│   │   │   │   ├── 🖼️ somdejHæwkman1front.png
│   │   │   │   └── 🖼️ somdejHæwkman6front.png
│   │   │   └── 📁 ออกวัดหนองอีดุก
│   │   │       ├── 🖼️ watnongEduk4front.png
│   │   │       ├── 🖼️ watnongEduk7back.png
│   │   │       └── 🖼️ watnongEduk9back.png
│   │   └── ⚙️ organization_report.json
│   ├── 📁 docs
│   │   └── 📄 README_ADVANCED.md
│   ├── 📁 evaluation
│   │   └── 🐍 test_emergency_model.py
│   ├── 📁 pipelines
│   │   ├── 🐍 advanced_data_pipeline.py
│   │   ├── 🐍 advanced_image_processor.py
│   │   └── 🐍 debug_data_pipeline.py
│   ├── 📁 saved_models
│   ├── 📁 training
│   │   ├── 🐍 master_training_system.py
│   │   ├── 🐍 memory_optimized_training.py
│   │   └── 🐍 ultra_simple_training.py
│   ├── 📁 training_output
│   │   ├── 📁 embeddings
│   │   │   └── 📎 embeddings.db
│   │   ├── 📁 logs
│   │   ├── 📁 models
│   │   ├── 📁 reports
│   │   │   ├── ⚙️ dataset_analysis.json
│   │   │   └── ⚙️ pipeline_stats.json
│   │   ├── 📁 tensorboard
│   │   │   ├── 📎 events.out.tfevents.1756821126.DESKTOP-0BE5LED.6068.0
│   │   │   ├── 📎 events.out.tfevents.1756821203.DESKTOP-0BE5LED.2060.0
│   │   │   ├── 📎 events.out.tfevents.1756822791.DESKTOP-0BE5LED.7248.0
│   │   │   ├── 📎 events.out.tfevents.1756823116.DESKTOP-0BE5LED.11952.0
│   │   │   ├── 📎 events.out.tfevents.1756823280.DESKTOP-0BE5LED.9060.0
│   │   │   ├── 📎 events.out.tfevents.1756823526.DESKTOP-0BE5LED.13584.0
│   │   │   ├── 📎 events.out.tfevents.1756823704.DESKTOP-0BE5LED.13748.0
│   │   │   ├── 📎 events.out.tfevents.1756824029.DESKTOP-0BE5LED.3084.0
│   │   │   ├── 📎 events.out.tfevents.1756824503.DESKTOP-0BE5LED.16316.0
│   │   │   ├── 📎 events.out.tfevents.1756824608.DESKTOP-0BE5LED.6724.0
│   │   │   ├── 📎 events.out.tfevents.1756824681.DESKTOP-0BE5LED.13780.0
│   │   │   ├── 📎 events.out.tfevents.1756824741.DESKTOP-0BE5LED.11396.0
│   │   │   ├── 📎 events.out.tfevents.1756824834.DESKTOP-0BE5LED.10836.0
│   │   │   ├── 📎 events.out.tfevents.1756824919.DESKTOP-0BE5LED.14900.0
│   │   │   ├── 📎 events.out.tfevents.1756825074.DESKTOP-0BE5LED.10332.0
│   │   │   ├── 📎 events.out.tfevents.1756829712.DESKTOP-0BE5LED.15856.0
│   │   │   └── 📎 events.out.tfevents.1756845927.DESKTOP-0BE5LED.13324.0
│   │   ├── 📁 visualizations
│   │   ├── ⚙️ config.json
│   │   ├── 📎 emergency_model.pth
│   │   ├── ⚙️ emergency_training_results.json
│   │   ├── 📄 EXECUTIVE_SUMMARY.txt
│   │   ├── ⚙️ FINAL_COMPREHENSIVE_REPORT.json
│   │   ├── ⚙️ PRODUCTION_MODEL_INFO.json
│   │   ├── 📎 step5_checkpoint_epoch_1.pth
│   │   ├── 📎 step5_checkpoint_epoch_2.pth
│   │   ├── 📎 step5_checkpoint_epoch_3.pth
│   │   ├── 📎 step5_final_model.pth
│   │   ├── ⚙️ step5_training_report.json
│   │   ├── ⚙️ test_results.json
│   │   ├── 📎 ultra_simple_model.pth
│   │   └── ⚙️ ultra_simple_training_report.json
│   ├── 🐍 __init__.py
│   ├── 🐍 dataset_organizer.py
│   ├── 🐍 final_steps_6_and_7.py
│   ├── 🐍 self_supervised_learning.py
│   └── 🐍 setup_advanced.py
├── 📁 archive
│   ├── 📁 launchers
│   │   ├── 🐍 launch_complete_system.py
│   │   ├── 🐍 launch_real_ai_system.py
│   │   ├── 🐍 launch_system.py
│   │   ├── 🐍 quick_start.py
│   │   ├── 🐍 simple_launcher.py
│   │   ├── 🐍 start.py
│   │   ├── 🐍 start_amulet_system.py
│   │   └── 🐍 system_launcher.py
│   ├── 📎 batches
│   ├── 📎 configs
│   ├── 📎 scripts
│   └── 📎 tests
├── 📁 backend
│   ├── 📁 api
│   │   ├── 🐍 api.py
│   │   ├── 🐍 api_with_real_model.py
│   │   └── 🐍 optimized_api.py
│   ├── 📁 config
│   │   └── 🐍 config.py
│   ├── 📁 models
│   │   ├── 🐍 model_loader.py
│   │   ├── 🐍 optimized_model_loader.py
│   │   └── 🐍 real_model_loader.py
│   ├── 📁 services
│   │   ├── 🐍 ai_model_service.py
│   │   ├── 🐍 market_scraper.py
│   │   ├── 🐍 price_estimator.py
│   │   ├── 🐍 recommend.py
│   │   ├── 🐍 recommend_optimized.py
│   │   ├── 🐍 similarity_search.py
│   │   └── 🐍 valuation.py
│   ├── 📁 tests
│   │   └── 🐍 test_api.py
│   └── 🐍 __init__.py
├── 📁 backups
│   └── 📁 backup_20250903-212636
│       ├── 📁 ai_models
│       │   └── ⚙️ labels.json
│       ├── 📁 backend
│       │   ├── 🐍 api.py
│       │   └── 🐍 model_loader.py
│       ├── 📁 docs
│       │   └── 📄 SYSTEM_GUIDE.md
│       ├── 📁 frontend
│       │   └── 🐍 app_streamlit.py
│       ├── 🐍 amulet_launcher.py
│       ├── ⚙️ config.json
│       ├── 📄 README.md
│       ├── 📄 requirements.txt
│       └── 🐍 setup_models.py
├── 📁 config
├── 📁 dataset
│   ├── 📁 somdej-fatherguay
│   │   ├── 🖼️ somdej-beginning-fatherguay-b1(bw).jpg
│   │   ├── 🖼️ somdej-beginning-fatherguay-b1.jpg
│   │   ├── 🖼️ somdej-beginning-fatherguay-f1(bw).jpg
│   │   ├── 🖼️ somdej-beginning-fatherguay-f1.jpg
│   │   ├── 🖼️ somdej-pongnammun-fatherguay-b1(bw).jpg
│   │   ├── 🖼️ somdej-pongnammun-fatherguay-b1.jpg
│   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1(bw).jpg
│   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1.jpg
│   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-b1(bw).jpg
│   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-b1.jpg
│   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-f1(bw).jpg
│   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-f1.jpg
│   │   ├── 🖼️ somdej-tekliengakising-fatherguay-b1(bw).jpg
│   │   ├── 🖼️ somdej-tekliengakising-fatherguay-b1.jpg
│   │   ├── 🖼️ somdej-tekliengakising-fatherguay-f1(bw).jpg
│   │   ├── 🖼️ somdej-tekliengakising-fatherguay-f1.jpg
│   │   ├── 🖼️ somdejbackjarnyan-fatherguay-b1(bw).jpg
│   │   ├── 🖼️ somdejbackjarnyan-fatherguay-b1.jpg
│   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1(bw).jpg
│   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1.jpg
│   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1(bw).jpg
│   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1.jpg
│   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2(bw).jpg
│   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2.jpg
│   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-b1(bw).jpg
│   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-b1.jpg
│   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-f1(bw).jpg
│   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-f1.jpg
│   │   ├── 🖼️ somdejsrewaree-fatherguay-b2(bw).jpg
│   │   ├── 🖼️ somdejsrewaree-fatherguay-b2.jpg
│   │   ├── 🖼️ somdejsrewaree-fatherguay-f1(bw).jpg
│   │   └── 🖼️ somdejsrewaree-fatherguay-f1.jpg
│   ├── 📁 พระพุทธเจ้าในวิหาร
│   │   ├── 🖼️ Phraphuthjao in viharn1front.png
│   │   ├── 🖼️ Phraphuthjao in viharn2front.png
│   │   ├── 🖼️ Phraphuthjao in viharn3front.png
│   │   ├── 🖼️ Phraphuthjao in viharn4front.png
│   │   ├── 🖼️ Phraphuthjao in viharn5front.png
│   │   ├── 🖼️ Phraphuthjao in viharn6back.png
│   │   └── 🖼️ Phraphuthjao in viharn7back.png
│   ├── 📁 พระสมเด็จฐานสิงห์
│   │   ├── 🖼️ somdejthansing 1back.png
│   │   ├── 🖼️ somdejthansing 1front.png
│   │   ├── 🖼️ somdejthansing 2back.png
│   │   └── 🖼️ somdejthansing 2front.png
│   ├── 📁 พระสมเด็จประทานพร พุทธกวัก
│   │   ├── 🖼️ Prasomdej-pudtagueg1-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg1-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg10-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg10-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg11-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg11-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg12-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg12-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg13-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg13-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg2-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg2-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg3-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg3-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg4-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg4-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg5-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg5-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg6-back.png
│   │   ├── 🖼️ Prasomdej-pudtagueg6-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg7-front.png
│   │   ├── 🖼️ Prasomdej-pudtagueg8-front.png
│   │   └── 🖼️ Prasomdej-pudtagueg9-front.png
│   ├── 📁 พระสมเด็จหลังรูปเหมือน
│   │   ├── 🖼️ prasomejhaunlouphmuan 10back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 10front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 11back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 11front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 12back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 12front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 13back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 13front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 1back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 1front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 2back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 2front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 3back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 3front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 4back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 4front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 5back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 5front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 6back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 6front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 7back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 7front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 8back.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 8front.png
│   │   ├── 🖼️ prasomejhaunlouphmuan 9back.png
│   │   └── 🖼️ prasomejhaunlouphmuan 9front.png
│   ├── 📁 พระสรรค์
│   │   ├── 🖼️ prasan 1back.png
│   │   └── 🖼️ prasan 1front.png
│   ├── 📁 พระสิวลี
│   │   ├── 🖼️ prasrewaree 10front.png
│   │   ├── 🖼️ prasrewaree 1back.png
│   │   ├── 🖼️ prasrewaree 1front.png
│   │   ├── 🖼️ prasrewaree 2back.png
│   │   ├── 🖼️ prasrewaree 2front.png
│   │   ├── 🖼️ prasrewaree 3back.png
│   │   ├── 🖼️ prasrewaree 3front.png
│   │   ├── 🖼️ prasrewaree 4back.png
│   │   ├── 🖼️ prasrewaree 4front.png
│   │   ├── 🖼️ prasrewaree 5back.png
│   │   ├── 🖼️ prasrewaree 5front.png
│   │   ├── 🖼️ prasrewaree 6back.png
│   │   ├── 🖼️ prasrewaree 6front.png
│   │   ├── 🖼️ prasrewaree 7front.png
│   │   ├── 🖼️ prasrewaree 8front.png
│   │   └── 🖼️ prasrewaree 9front.png
│   ├── 📁 สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ
│   │   ├── 🖼️ somdejpimprougpo9bai 10back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 10front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 1back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 1front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 2back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 2front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 3back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 3front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 4back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 4front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 5back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 5front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 6back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 6front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 7back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 7front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 8back.png
│   │   ├── 🖼️ somdejpimprougpo9bai 8front.png
│   │   ├── 🖼️ somdejpimprougpo9bai 9back.png
│   │   └── 🖼️ somdejpimprougpo9bai 9front.png
│   ├── 📁 สมเด็จแหวกม่าน
│   │   ├── 🖼️ somdejHæwkman10front.png
│   │   ├── 🖼️ somdejHæwkman11front.png
│   │   ├── 🖼️ somdejHæwkman12front.png
│   │   ├── 🖼️ somdejHæwkman13front.png
│   │   ├── 🖼️ somdejHæwkman14front.png
│   │   ├── 🖼️ somdejHæwkman15front.png
│   │   ├── 🖼️ somdejHæwkman16front.png
│   │   ├── 🖼️ somdejHæwkman17front.png
│   │   ├── 🖼️ somdejHæwkman18front.png
│   │   ├── 🖼️ somdejHæwkman19front.png
│   │   ├── 🖼️ somdejHæwkman1front.png
│   │   ├── 🖼️ somdejHæwkman20front.png
│   │   ├── 🖼️ somdejHæwkman21front.png
│   │   ├── 🖼️ somdejHæwkman2front.png
│   │   ├── 🖼️ somdejHæwkman3front.png
│   │   ├── 🖼️ somdejHæwkman4front.png
│   │   ├── 🖼️ somdejHæwkman5front.png
│   │   ├── 🖼️ somdejHæwkman6front.png
│   │   ├── 🖼️ somdejHæwkman7front.png
│   │   ├── 🖼️ somdejHæwkman8front.png
│   │   └── 🖼️ somdejHæwkman9front.png
│   └── 📁 ออกวัดหนองอีดุก
│       ├── 🖼️ watnongEduk10back.png
│       ├── 🖼️ watnongEduk10front.png
│       ├── 🖼️ watnongEduk1back.png
│       ├── 🖼️ watnongEduk1front.png
│       ├── 🖼️ watnongEduk2back.png
│       ├── 🖼️ watnongEduk2front.png
│       ├── 🖼️ watnongEduk3back.png
│       ├── 🖼️ watnongEduk3front.png
│       ├── 🖼️ watnongEduk4back.png
│       ├── 🖼️ watnongEduk4front.png
│       ├── 🖼️ watnongEduk5back.png
│       ├── 🖼️ watnongEduk5front.png
│       ├── 🖼️ watnongEduk6front.png
│       ├── 🖼️ watnongEduk7back.png
│       ├── 🖼️ watnongEduk7front.png
│       ├── 🖼️ watnongEduk8back.png
│       ├── 🖼️ watnongEduk8front.png
│       ├── 🖼️ watnongEduk9back.png
│       └── 🖼️ watnongEduk9front.png
├── 📁 dataset_organized
│   ├── 📁 somdej_fatherguay
│   ├── 📁 somdej_portrait_back
│   ├── 📁 somdej_prok_bodhi
│   ├── 📁 somdej_waek_man
│   ├── 📁 wat_nong_e_duk
│   ├── 📁 wat_nong_e_duk_misc
│   ├── ⚙️ labels.json
│   └── ⚙️ labels_karaoke.json
├── 📁 dataset_split
│   ├── 📁 test
│   │   ├── 📁 somdej-fatherguay
│   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1.jpg
│   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1.jpg
│   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1(bw).jpg
│   │   │   └── 🖼️ somdejsrewaree-fatherguay-b2(bw).jpg
│   │   ├── 📁 พระพุทธเจ้าในวิหาร
│   │   │   ├── 🖼️ Phraphuthjao in viharn1front.png
│   │   │   └── 🖼️ Phraphuthjao in viharn3front.png
│   │   ├── 📁 พระสมเด็จฐานสิงห์
│   │   │   ├── 🖼️ somdejthansing 1front.png
│   │   │   └── 🖼️ somdejthansing 2back.png
│   │   ├── 📁 พระสมเด็จประทานพร พุทธกวัก
│   │   │   ├── 🖼️ Prasomdej-pudtagueg12-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg3-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg3-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg4-back.png
│   │   │   └── 🖼️ Prasomdej-pudtagueg8-front.png
│   │   ├── 📁 พระสมเด็จหลังรูปเหมือน
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 1back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 2back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 3front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 5front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 7front.png
│   │   │   └── 🖼️ prasomejhaunlouphmuan 8front.png
│   │   ├── 📁 พระสรรค์
│   │   │   └── 🖼️ prasan 1back.png
│   │   ├── 📁 พระสิวลี
│   │   │   ├── 🖼️ prasrewaree 10front.png
│   │   │   ├── 🖼️ prasrewaree 3front.png
│   │   │   ├── 🖼️ prasrewaree 4front.png
│   │   │   └── 🖼️ prasrewaree 6back.png
│   │   ├── 📁 สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ
│   │   │   ├── 🖼️ somdejpimprougpo9bai 10front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 1back.png
│   │   │   └── 🖼️ somdejpimprougpo9bai 8back.png
│   │   ├── 📁 สมเด็จแหวกม่าน
│   │   │   ├── 🖼️ somdejHæwkman16front.png
│   │   │   ├── 🖼️ somdejHæwkman1front.png
│   │   │   ├── 🖼️ somdejHæwkman20front.png
│   │   │   ├── 🖼️ somdejHæwkman7front.png
│   │   │   └── 🖼️ somdejHæwkman8front.png
│   │   └── 📁 ออกวัดหนองอีดุก
│   │       ├── 🖼️ watnongEduk10back.png
│   │       ├── 🖼️ watnongEduk2front.png
│   │       ├── 🖼️ watnongEduk8back.png
│   │       ├── 🖼️ watnongEduk8front.png
│   │       └── 🖼️ watnongEduk9front.png
│   ├── 📁 train
│   │   ├── 📁 somdej-fatherguay
│   │   │   ├── 🖼️ somdej-beginning-fatherguay-b1(bw).jpg
│   │   │   ├── 🖼️ somdej-beginning-fatherguay-b1.jpg
│   │   │   ├── 🖼️ somdej-beginning-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdej-beginning-fatherguay-f1.jpg
│   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-b1(bw).jpg
│   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-b1.jpg
│   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdej-pongnammun-fatherguay-f1.jpg
│   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-b1(bw).jpg
│   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-b1.jpg
│   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-f1.jpg
│   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-b1(bw).jpg
│   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-b1.jpg
│   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdej-tekliengakising-fatherguay-f1.jpg
│   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-b1(bw).jpg
│   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-b1.jpg
│   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1.jpg
│   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1(bw).jpg
│   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1.jpg
│   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2(bw).jpg
│   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2.jpg
│   │   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-b1(bw).jpg
│   │   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-b1.jpg
│   │   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdejkatohe-pongnammun-fatherguay-f1.jpg
│   │   │   ├── 🖼️ somdejsrewaree-fatherguay-b2(bw).jpg
│   │   │   ├── 🖼️ somdejsrewaree-fatherguay-b2.jpg
│   │   │   ├── 🖼️ somdejsrewaree-fatherguay-f1(bw).jpg
│   │   │   └── 🖼️ somdejsrewaree-fatherguay-f1.jpg
│   │   ├── 📁 พระพุทธเจ้าในวิหาร
│   │   │   ├── 🖼️ Phraphuthjao in viharn1front.png
│   │   │   ├── 🖼️ Phraphuthjao in viharn2front.png
│   │   │   ├── 🖼️ Phraphuthjao in viharn3front.png
│   │   │   ├── 🖼️ Phraphuthjao in viharn4front.png
│   │   │   ├── 🖼️ Phraphuthjao in viharn5front.png
│   │   │   ├── 🖼️ Phraphuthjao in viharn6back.png
│   │   │   └── 🖼️ Phraphuthjao in viharn7back.png
│   │   ├── 📁 พระสมเด็จฐานสิงห์
│   │   │   ├── 🖼️ somdejthansing 1back.png
│   │   │   ├── 🖼️ somdejthansing 1front.png
│   │   │   ├── 🖼️ somdejthansing 2back.png
│   │   │   └── 🖼️ somdejthansing 2front.png
│   │   ├── 📁 พระสมเด็จประทานพร พุทธกวัก
│   │   │   ├── 🖼️ Prasomdej-pudtagueg1-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg1-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg10-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg10-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg11-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg11-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg12-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg13-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg13-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg2-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg2-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg3-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg3-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg4-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg4-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg5-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg5-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg6-back.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg6-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg7-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg8-front.png
│   │   │   └── 🖼️ Prasomdej-pudtagueg9-front.png
│   │   ├── 📁 พระสมเด็จหลังรูปเหมือน
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 10front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 11back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 11front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 12back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 12front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 13back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 13front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 1back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 1front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 2back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 2front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 3back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 3front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 4back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 4front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 5back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 5front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 6back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 6front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 7back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 7front.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 8back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 9back.png
│   │   │   └── 🖼️ prasomejhaunlouphmuan 9front.png
│   │   ├── 📁 พระสรรค์
│   │   │   ├── 🖼️ prasan 1back.png
│   │   │   └── 🖼️ prasan 1front.png
│   │   ├── 📁 พระสิวลี
│   │   │   ├── 🖼️ prasrewaree 10front.png
│   │   │   ├── 🖼️ prasrewaree 1back.png
│   │   │   ├── 🖼️ prasrewaree 1front.png
│   │   │   ├── 🖼️ prasrewaree 2back.png
│   │   │   ├── 🖼️ prasrewaree 2front.png
│   │   │   ├── 🖼️ prasrewaree 3back.png
│   │   │   ├── 🖼️ prasrewaree 3front.png
│   │   │   ├── 🖼️ prasrewaree 4back.png
│   │   │   ├── 🖼️ prasrewaree 4front.png
│   │   │   ├── 🖼️ prasrewaree 5back.png
│   │   │   ├── 🖼️ prasrewaree 5front.png
│   │   │   ├── 🖼️ prasrewaree 6back.png
│   │   │   ├── 🖼️ prasrewaree 6front.png
│   │   │   ├── 🖼️ prasrewaree 7front.png
│   │   │   ├── 🖼️ prasrewaree 8front.png
│   │   │   └── 🖼️ prasrewaree 9front.png
│   │   ├── 📁 สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ
│   │   │   ├── 🖼️ somdejpimprougpo9bai 10back.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 10front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 1front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 2back.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 2front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 3back.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 3front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 4back.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 4front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 5back.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 5front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 6back.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 6front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 7back.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 7front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 8front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 9back.png
│   │   │   └── 🖼️ somdejpimprougpo9bai 9front.png
│   │   ├── 📁 สมเด็จแหวกม่าน
│   │   │   ├── 🖼️ somdejHæwkman10front.png
│   │   │   ├── 🖼️ somdejHæwkman11front.png
│   │   │   ├── 🖼️ somdejHæwkman12front.png
│   │   │   ├── 🖼️ somdejHæwkman13front.png
│   │   │   ├── 🖼️ somdejHæwkman14front.png
│   │   │   ├── 🖼️ somdejHæwkman15front.png
│   │   │   ├── 🖼️ somdejHæwkman17front.png
│   │   │   ├── 🖼️ somdejHæwkman18front.png
│   │   │   ├── 🖼️ somdejHæwkman19front.png
│   │   │   ├── 🖼️ somdejHæwkman20front.png
│   │   │   ├── 🖼️ somdejHæwkman21front.png
│   │   │   ├── 🖼️ somdejHæwkman2front.png
│   │   │   ├── 🖼️ somdejHæwkman3front.png
│   │   │   ├── 🖼️ somdejHæwkman4front.png
│   │   │   ├── 🖼️ somdejHæwkman5front.png
│   │   │   ├── 🖼️ somdejHæwkman6front.png
│   │   │   ├── 🖼️ somdejHæwkman7front.png
│   │   │   ├── 🖼️ somdejHæwkman8front.png
│   │   │   └── 🖼️ somdejHæwkman9front.png
│   │   └── 📁 ออกวัดหนองอีดุก
│   │       ├── 🖼️ watnongEduk10back.png
│   │       ├── 🖼️ watnongEduk10front.png
│   │       ├── 🖼️ watnongEduk1back.png
│   │       ├── 🖼️ watnongEduk1front.png
│   │       ├── 🖼️ watnongEduk2back.png
│   │       ├── 🖼️ watnongEduk2front.png
│   │       ├── 🖼️ watnongEduk3back.png
│   │       ├── 🖼️ watnongEduk3front.png
│   │       ├── 🖼️ watnongEduk4back.png
│   │       ├── 🖼️ watnongEduk4front.png
│   │       ├── 🖼️ watnongEduk5back.png
│   │       ├── 🖼️ watnongEduk5front.png
│   │       ├── 🖼️ watnongEduk6front.png
│   │       ├── 🖼️ watnongEduk7back.png
│   │       ├── 🖼️ watnongEduk7front.png
│   │       ├── 🖼️ watnongEduk8back.png
│   │       ├── 🖼️ watnongEduk8front.png
│   │       ├── 🖼️ watnongEduk9back.png
│   │       └── 🖼️ watnongEduk9front.png
│   ├── 📁 validation
│   │   ├── 📁 somdej-fatherguay
│   │   │   ├── 🖼️ somdej-pongnammunyanraw5-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdejbackjarnyan-fatherguay-f1(bw).jpg
│   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f1.jpg
│   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2(bw).jpg
│   │   │   ├── 🖼️ somdejblockhinmerdkon-pongguay-meatvansoapblood-f2.jpg
│   │   │   └── 🖼️ somdejsrewaree-fatherguay-f1.jpg
│   │   ├── 📁 พระพุทธเจ้าในวิหาร
│   │   │   └── 🖼️ Phraphuthjao in viharn4front.png
│   │   ├── 📁 พระสมเด็จประทานพร พุทธกวัก
│   │   │   ├── 🖼️ Prasomdej-pudtagueg4-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg5-front.png
│   │   │   ├── 🖼️ Prasomdej-pudtagueg7-front.png
│   │   │   └── 🖼️ Prasomdej-pudtagueg9-front.png
│   │   ├── 📁 พระสมเด็จหลังรูปเหมือน
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 10back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 6back.png
│   │   │   ├── 🖼️ prasomejhaunlouphmuan 8front.png
│   │   │   └── 🖼️ prasomejhaunlouphmuan 9back.png
│   │   ├── 📁 พระสิวลี
│   │   │   ├── 🖼️ prasrewaree 5front.png
│   │   │   ├── 🖼️ prasrewaree 6front.png
│   │   │   └── 🖼️ prasrewaree 8front.png
│   │   ├── 📁 สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ
│   │   │   ├── 🖼️ somdejpimprougpo9bai 1back.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 3front.png
│   │   │   ├── 🖼️ somdejpimprougpo9bai 7front.png
│   │   │   └── 🖼️ somdejpimprougpo9bai 9front.png
│   │   ├── 📁 สมเด็จแหวกม่าน
│   │   │   ├── 🖼️ somdejHæwkman10front.png
│   │   │   ├── 🖼️ somdejHæwkman16front.png
│   │   │   ├── 🖼️ somdejHæwkman1front.png
│   │   │   └── 🖼️ somdejHæwkman6front.png
│   │   └── 📁 ออกวัดหนองอีดุก
│   │       ├── 🖼️ watnongEduk4front.png
│   │       ├── 🖼️ watnongEduk7back.png
│   │       └── 🖼️ watnongEduk9back.png
│   └── ⚙️ organization_report.json
├── 📁 docs
│   ├── 📁 api
│   │   └── 📄 API.md
│   ├── 📁 development
│   │   └── 📄 DEPLOYMENT.md
│   ├── 📁 guides
│   ├── 📁 system
│   │   ├── 📄 SYSTEM_GUIDE.md
│   │   └── 📄 SYSTEM_GUIDE_updated.md
│   ├── 📄 CHANGELOG.md
│   ├── 📄 DIRECTORY_STRUCTURE.md
│   ├── 📄 MODULAR_ARCHITECTURE.md
│   └── 📄 PROJECT_STRUCTURE.md
├── 📁 frontend
│   ├── 📁 assets
│   ├── 📁 components
│   ├── 📁 pages
│   ├── 📁 utils
│   │   └── 🐍 utils.py
│   ├── 🐍 __init__.py
│   └── 🐍 app_streamlit.py
├── 📁 logs
├── 📁 scripts
│   ├── 🐍 amulet_launcher.py
│   ├── 🐍 main_launcher.py
│   ├── 🐍 setup_complete_system.py
│   ├── � setup_models.py
│   ├── �🔧 amulet_launcher.bat
│   ├── 🔧 start.bat
│   ├── 🔧 launch.bat
│   ├── 🔧 organize.bat
│   ├── 🔧 organize_folders.bat
│   ├── 🔧 initialize_structure.bat
│   └── 🐍 test_system.py
├── 📁 tests
│   ├── 📁 data
│   ├── 📁 fixtures
│   │   └── 🐍 conftest.py
│   ├── 📁 integration
│   ├── 📁 test_images
│   ├── 📁 unit
│   ├── 🐍 test_api.py
│   ├── 🐍 test_config_manager.py
│   ├── 🐍 test_file_operations.py
│   └── 📄 test_write.txt
├── 📁 tools
│   ├── 🐍 __init__.py
│   ├── 🐍 amulet_toolkit.py
│   ├── 🐍 cleanup.py
│   ├── 🐍 cleanup_files_phase2.py
│   ├── 🐍 cleanup_root.py
│   ├── 🐍 comprehensive_file_test.py
│   ├── 🐍 file_access_test.py
│   ├── 🐍 maintenance.py
│   ├── 🐍 organize_files.py
│   ├── 🐍 organize_internal_structure.py
│   ├── � repair_system.py
│   ├── 🐍 restructure_project.py
│   ├── 🐍 verify_system.py
│   ├── 📄 README.md
│   └── 🔧 show_project_structure.ps1
├── 📁 training_output
│   ├── 📁 embeddings
│   │   └── 📎 embeddings.db
│   ├── 📁 logs
│   ├── 📁 models
│   ├── 📁 reports
│   │   ├── ⚙️ dataset_analysis.json
│   │   └── ⚙️ pipeline_stats.json
│   ├── 📁 tensorboard
│   │   ├── 📎 events.out.tfevents.1756826357.DESKTOP-0BE5LED.4076.0
│   │   ├── 📎 events.out.tfevents.1756827794.DESKTOP-0BE5LED.4700.0
│   │   └── 📎 events.out.tfevents.1756828173.DESKTOP-0BE5LED.12956.0
│   ├── 📁 visualizations
│   └── ⚙️ config.json
├── 📁 utils
│   ├── 📁 config
│   │   └── 🐍 config_manager.py
│   ├── 📁 data
│   ├── 📁 image
│   │   └── 🐍 image_utils.py
│   ├── 📁 logging
│   │   ├── 🐍 error_handler.py
│   │   └── 🐍 logger.py
│   └── 🐍 __init__.py
├── ⚙️ config.json
├──  README.md
├── 📄 requirements.txt
├── � cleanup_root.bat
└── (All other files have been organized into appropriate folders)
```

## Main Directory Descriptions

- **ai_models/**: Contains AI model files, training scripts, and model-related utilities
  - **core/**: Core model files and labels
  - **training/**: Training scripts and modules
  - **pipelines/**: Data processing pipelines
  - **evaluation/**: Model testing and evaluation
  - **configs/**: Model configurations
  - **docs/**: Model documentation

- **backend/**: Backend API and server code
  - **api/**: API endpoints and interfaces
  - **models/**: Model loading and processing
  - **services/**: Backend services (valuation, recommendations, etc.)
  - **config/**: Backend configuration
  - **tests/**: Backend tests

- **frontend/**: Frontend UI code
  - **pages/**: Main pages and views
  - **components/**: Reusable UI components
  - **utils/**: Frontend utilities

- **docs/**: Documentation files
  - **api/**: API documentation
  - **guides/**: User and developer guides
  - **system/**: System architecture and design
  - **development/**: Development and deployment guides

- **scripts/**: Launch and utility scripts
- **tests/**: Test files
  - **unit/**: Unit tests
  - **integration/**: Integration tests
  - **fixtures/**: Test fixtures
  - **data/**: Test data

- **tools/**: Maintenance and utility tools
- **utils/**: Utility functions and modules
  - **config/**: Configuration utilities
  - **image/**: Image processing utilities
  - **logging/**: Logging and error handling
  - **data/**: Data processing utilities

## File Consolidation

Several files were consolidated to reduce redundancy:

1. **utils/utils.py**: Combined utility functions from multiple source files
2. **backend/models.py**: Consolidated model loader implementations
3. **backend/api.py**: Unified API implementations

## Migration Changes

Original files that were moved:
- Move test files to tests directory
- Move launcher scripts to scripts directory
- Move configuration files to config directory
- Move documentation to docs directory
- Move API documentation to docs/api directory
- Move guides to docs/guides directory
- Move deprecated files to archive directory

## Internal Folder Organization

The following folders were organized with more detailed structure:
- Organize AI models directory
- Organize backend directory
- Organize frontend directory
- Organize documentation directory
- Organize utilities directory
- Organize tests directory
