import cv2 # OpenCV kütüphanesini içeri aktarıyoruz.
import numpy as np # Numpy kütüphanesini içeri aktarıyoruz.
import time # time kütüphanesini içeri aktarıyoruz.


def calculate_measures_4roi(frame): # calculate_measures_4roi fonksiyonu oluşturduk.
   
    height = frame.shape[0] # Aldığı frame'in yüksekliğini alıyoruz.
    width = frame.shape[1]  # Genişliğini alıyoruz.

    corners4roi = [ # ilgilendiğimiz alanın 4 köşeseni belirliyoruz.
        (0, height), # 1. köşenin en sol alttaki koordinatını veriyoruz.
        (width / 3.2, height / 1.8),
        (width / 1.5, height / 1.5),
        (width, height)]
    
    return corners4roi # köşe değerlerini bu fonksiyonun çıktısı olarak döndürüyoruz.


def region_of_interest(frame, corners4roi):    # İlgilendiğimiz alanın maskeleme
    # işlemlerini yapmak için fonksiyon kullanıyoruz.
    mask = np.zeros_like(frame) # boş bir resim oluşturuyoruz görüntümüzün boyutunda
    #channel_count = img.shape[2]
    match_mask_color = 255 # beyaz rengi maskeliyoruz.
    cv2.fillPoly(mask, corners4roi, match_mask_color) 
    masked_frame = cv2.bitwise_and(frame, mask) # Bitwise fonksiyonu kullanarak maskelenmiş frame'i değişkene atıyoruz.
    
    return masked_frame # Maskelenmiş frame'i döndürüyoruz.


def drow_the_lines(frame, lines): # şerit çizgilerinin üzerinde renkli çizgi çizdiren fonksiyon oluşturduk.
    
    frame = np.copy(frame)  # frame'i kopyalıyoruz.
    blank_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)   # Orijinal resim ile aynı büyuklükte boş bir resim oluşturuyoruz.

    for line in lines:      # Bu for döngüsü ile daha önce oluşturduğumuz boş resimin içine tespit ettiğimiz tüm şerit çizgilerini ekliyoruz.
        for x1, y1, x2, y2 in line:
            cv2.line(blank_frame, (x1,y1), (x2,y2), (0, 255, 0), thickness=5)    ## (0,255,0) yeşil renk, 5 kalınlıkta (x1,y1)'den
            # (x2,y2)'ye kadar çizgiyi çizdiriyoruz.

    frame = cv2.addWeighted(frame, 0.9, blank_frame, 0.5, 0.0)    # Orijinal frame ile şerit çizgilerini içeren blank frame üst üste ekleniyor. ağırlıklı (0.8) birleştirme uygulanıyor.
    
    return frame # frame değerini döndürüyoruz.


def detect_lines_and_process(frame): # çizgi algılama ve tüm frame işleme işlemlerini yapacağımız
    # fonksiyonumuzu oluşturuyoruz.
    blurred_frame = cv2.GaussianBlur(frame,(9,9),0.8)   # Frame'mizdeki keskinlikleri azaltmak için blurlama yapıyoruz.
    gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2GRAY)   # Resmi gri ölçeğe dönüştürme
    canny_frame = cv2.Canny(gray_frame, 100, 120) # Canny fonksiyonu ile iki eşik değeri (100,120) arasında kenar çizgileri algılatıyoruz (edge detection)
    cropped_frame = region_of_interest(canny_frame,
                    np.array([calculate_measures_4roi(frame)], np.int32),)
    lines = cv2.HoughLinesP(cropped_frame,        # HoughLinesP transform ile probabilistik (olasılıksal) dönüştürme yapılıyor.
                            rho=2,                # burada HoughLinesP transform ile resimdeki çizgiler (şerit çizgileri) tanımlanmış oluyor. 
                            theta=np.pi/180,
                            threshold=135,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    frame_with_lines = drow_the_lines(frame, lines)
    
    return frame_with_lines # çizgiler çizilmiş frame'mizi döndürüyoruz.



cap = cv2.VideoCapture("Şerit Takip Sistemi/Road1.mp4")    # Videomuzu okuyoruz.


while cap.isOpened():   # Videomuz okunduğu sürece dönecek döngü oluşturuyoruz.
    ret, frame = cap.read() # videomuzun her bir karesini frame değişkenimize atıyoruz.
    # ret = videonun başarılı okunup okunmadığını gösteriyor.
    cv2.imshow("Original Road", frame) # Orijinal yolumuzu ekrana yansıtıyoruz.
    
    frame = detect_lines_and_process(frame) # yolumuzun karesini işliyoruz.
    time.sleep(0.01) # video biraz yavaş ilerlesin diye 0.01 saniye programı uyutuyoruz.
    cropped_frame = region_of_interest(frame, # belirlediğimiz ölçülerde karemizi kırpıyoruz.
                                   np.array([calculate_measures_4roi(frame)], np.int32),)
    
    cv2.imshow("Cropped Road", cropped_frame) # Kırpılmış ve maskelenmiş kareyi ekrana yansıtıyoruz.
    
    cv2.imshow("Processed Road", frame) # işlenmiş karemizi Processed Road isimli pencerede açıyoruz.
    
    if cv2.waitKey(1) & 0xFF == ord("q"): # q tuşuna bastığımızda programdan çıkışı sağlıyor.
            break # çıkış anahtar kelimemiz break.
    
    
cap.release()
cv2.destroyAllWindows()