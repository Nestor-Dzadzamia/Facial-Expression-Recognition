# Representation Learning: Facial Expression Recognition Challenge

Kaggle competition : [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)

---

## პროექტის აღწერა

გვაქვს FER2013 მონაცემთა ბაზა, რომელიც შეიცავს (48x48 პიქსელის ზომის) სახის ნაცრისფერ სურათებს. სურათებზე მოცემულია ემოციების 7 კლასი:
`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`

- სატრენინგო სურათები: 28,709
- სატესტო სურათები: 7,178
- მიზანი: სახის გამომეტყველების კლასიფიკაცია

---

## პროექტის სტრუქტურა

```
├── notebooks/
│   ├── data_exploration.ipynb       
│   ├── baseline_cnn.ipynb          
│   ├── BatchNorm_Dropout_CNN.ipynb  
│   └── Augmented_CNN.ipynb          
└── README.md
```
---

## მონაცემთა ანალიზი

მონაცემში ვხედავთ კლასების დისბალანსს:

| ემოცია | სურათების რაოდენობა |
|--------|-------------------|
| Happy | 7,215 |
| Sad | 4,830 |
| Neutral | 4,965 |
| Fear | 4,097 |
| Angry | 3,995 |
| Surprise | 3,171 |
| Disgust | 436 |

`Disgust` კლასი მნიშვნელოვნად ნაკლებად არის წარმოდგენილი სხვებთან შედარებით.

---

## არქიტექტურები და ექსპერიმენტები

### არქიტექტურა 1 : Baseline CNN (`baseline_cnn.ipynb`)

**ჩემი მიდგომები:**
- 2 Convolutional Layer (16→32 ფილტრი)
- ReLU აქტივაცია და MaxPooling თითოეული კონვოლუციის შემდეგ
- 1 სრულად დაკავშირებული ლეიერი (128 ნეირონი)
- რეგულარიზაცია არ გამომიყენებია, რადგან მიზანი იყო საბაზისო შედეგის და overfitting-ის ჩვენება

**რატომ Baseline CNN:**
დავიწყე რაც შეიძლება მარტივი მოდელით, რათა გვენახა საწყისი შედეგი რეგულარიზაციის გარეშე.

**შედეგები (20 Epoch):**

| Metric | Percentage |
|---------|------------|
| Train Accuracy | 91.4% |
| Val Accuracy | 50.2% |
| Gap | 41.2% |

**ანალიზი:**
მოდელმა გამოავლინა კლასიკური overfitting. Train Accuracy 91%-ია, ხოლო Validation Accuracy 50%-ზე გაჩერდა. ვალიდაციის loss-მა კლება მე-6 ეპოქის შემდეგ შეაჩერა და ზრდა დაიწყო, სატრენინგო loss-ი კი კლებას განაგრძობდა. მიზეზი: მოდელმა დაიმახსოვრა სატრენინგო სურათები ზოგადი feature-ების სწავლის ნაცვლად.

---

### არქიტექტურა 2 : BatchNorm + Dropout CNN (`BatchNorm_Dropout_CNN.ipynb`)

**ჩემი მიდგომები:**
- 3 Convolutional Layer (32→64→128 ფილტრი)
- BatchNorm2d თითოეული კონვოლუციის შემდეგ
- Dropout(0.5)
- Fully Connected Layer (256 ნეირონი)

**შედეგები (30 Epoch):**

| Metric | Percentage |
|---------|------------|
| Train Accuracy | 66.7% |
| Val Accuracy | 54.9% |
| Gap | 11.8% |

**ანალიზი:**
რეგულარიზაციამ მნიშვნელოვნად შეამცირა overfitting. train/val გაპი 41.2%-დან 11.8%-მდე შემცირდა. ვალიდაციის სიზუსტე 50.2%-დან 54.9%-მდე გაიზარდა. სატრენინგო სიზუსტე baseline-ზე დაბალია (66.7%), ნორმალურია Dropout-მა ტრენინგი გაართულა.

---

### არქიტექტურა 3 : Augmented CNN (`Augmented_CNN.ipynb`)

**ჩემი მიდგომები:**
იგივე არქიტექტურაა, ერთადერთი ცვლილება არის მონაცემთა augmentation სატრენინგო pipeline-ში.

ამ notebook-ში 2 run გავუშვი:
- `augmented-cnn-no-aug` : augmentation-ის გარეშე (overfitting-ის ჩვენება)
- `augmented-cnn-with-aug` : augmentation-ით (გაუმჯობესება)

**Augmentation-ის ნაწილი**
- `RandomHorizontalFlip` : სახე ჰორიზონტალურად გადაბრუნებული (არეკლილი, ემოცია მაინც იგივეა)
- `RandomRotation(10°)` : მცირე დახრა, ვინაიდან სახეები ყოველთვის სრულყოფილად პირდაპირი არ არის
- `RandomCrop(48, padding=4)` : Random Cropping, ჯერ პადინგით გაიზრდება შემდეგ რენდომზე ამოიჭრება (48x48)

**რატომ ვიყენებ augmentation-ს:**
ტრენინგის დროს ყოველ Epoch-ში მოდელი  სურათის შემთხვევით შეცვლილ ვერსიას ხედავს. ეს დაზეპირებას შეუძლებელს ხდის და აიძულებს მოდელს ისწავლოს ზოგადი feature-ები.

**შედეგები (30 Epoch):**

| Run | Train Acc | Val Acc | Gap |
|-----|-----------|---------|------|
| No Augmentation | 73.5% | 55.5% | 17.9% |
| With Augmentation | 49.6% | **55.8%** | **0.8%** |

**ანალიზი:**
augmentation-მა train/val გაპი 17.9%-დან 0.8%-მდე დაიყვანა. overfitting აღარ არის. ვალიდაციის loss (1.14) ყველა run-ს შორის ყველაზე დაბალია.

---

## საერთო შედეგების შედარება

| არქიტექტურა | Train Acc | Val Acc | გაპი | WandB Run |
|------------|-----------|---------|------|-----------|
| Baseline CNN | 91.4% | 50.2% | 41.2% | baseline-cnn |
| BatchNorm+Dropout | 66.7% | 54.9% | 11.8% | batchnorm-dropout-cnn |
| No Augmentation | 73.5% | 55.5% | 17.9% | augmented-cnn-no-aug |
| With Augmentation | 49.6% | **55.8%** | **0.8%** | augmented-cnn-with-aug |

---

## დასკვნა

1. **Baseline** : რეგულარიზაციის გარეშე მოდელი სწრაფად იწყებს სატრენინგო მონაცემების დაზეპირებას. train/val გაპი 41% ანუ overfitting.

2. **BatchNorm + Dropout** : რეგულარიზაციამ გაპი 41%-დან 12%-მდე შეამცირა. ვალიდაციის სიზუსტე გაიზარდა.

3. **Data Augmentation** : ყველაზე კარგი მეთოდი overfitting-ს მოსაშორებლად. გაპი 0.8%. მოდელი ვერ ახერხებს დაზეპირებას, რადგან ყოველ ეპოქაში სხვადასხვა ვარიაციას ხედავს.

---


