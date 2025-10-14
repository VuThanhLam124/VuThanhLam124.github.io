---
title: "Normalizing Flow: Hành trình biến đổi từ Đơn giản đến Phức tạp"
date: "2025-01-15"
category: "flow-based-models"
tags: ["normalizing-flows", "generative-models", "beginner-guide", "deep-learning"]
description: "Một bài hướng dẫn thân thiện cho người mới bắt đầu về Normalizing Flow. Khám phá 'flow' là gì, tại sao chúng ta cần nó, và cách nó hoạt động qua các ví dụ trực quan."
---

# Normalizing Flow: Hành trình biến đổi từ Đơn giản đến Phức tạp

Chào mừng bạn đến với bài viết đầu tiên trong series về các mô hình sinh (Generative Models) dựa trên dòng chảy. Thay vì đi thẳng vào toán học khô khan, chúng ta hãy bắt đầu bằng một câu chuyện.

## Mục lục
1. [Câu chuyện về người thợ gốm: "Flow" là gì?](#1-câu-chuyện-về-người-thợ-gốm-flow-là-gì)
2. [Bài toán của AI: Tại sao cần một "người thợ gốm" giỏi?](#2-bài-toán-của-ai-tại-sao-cần-một-người-thợ-gốm-giỏi)
3. [Normalizing Flow: Người thợ gốm vừa khéo tay, vừa minh bạch](#3-normalizing-flow-người-thợ-gốm-vừa-khéo-tay-vừa-minh-bạch)
4. [Phép màu toán học: Làm sao để theo dõi sự biến đổi?](#4-phép-màu-toán-học-làm-sao-để-theo-dõi-sự-biến-đổi)
    - [4.1. Công thức Biến đổi Biến số (Change of Variables)](#41-công-thức-biến-đổi-biến-số-change-of-variables)
    - [4.2. "Phí co giãn": Vai trò của Định thức Jacobian](#42-phí-co-giãn-vai-trò-của-định-thức-jacobian)
5. [Thách thức thực tế: Vấn đề của dữ liệu lớn](#5-thách-thức-thực-tế-vấn-đề-của-dữ-liệu-lớn)
6. [Continuous Normalizing Flow: Từ rời rạc đến liên tục](#6-continuous-normalizing-flow-từ-rời-rạc-đến-liên-tục)
7. [Tổng kết và bước tiếp theo](#7-tổng-kết-và-bước-tiếp-theo)

---

## 1. Câu chuyện về người thợ gốm: "Flow" là gì?

Hãy tưởng tượng bạn là một người thợ gốm. Nhiệm vụ của bạn là tạo ra những chiếc bình gốm có hình dạng phức tạp (ví dụ: hình ngôi sao, hình con mèo). Tuy nhiên, bạn chỉ được bắt đầu với một khối đất sét hình cầu đơn giản.

Quá trình bạn nhào nặn, kéo, đẩy, và biến đổi khối đất sét hình cầu đó thành hình ngôi sao chính là một **"flow"** (dòng chảy). Đó là một chuỗi các phép biến đổi liên tiếp.

Quan trọng hơn, nếu bạn là một người thợ giỏi, bạn có thể làm ngược lại: biến chiếc bình hình ngôi sao trở lại thành khối cầu ban đầu. Quá trình biến đổi này có thể **đảo ngược (invertible)**.

Trong AI, "flow" cũng có ý nghĩa tương tự:
> **Flow** là một chuỗi các phép biến đổi toán học giúp chuyển một phân phối xác suất đơn giản (khối đất sét hình cầu) thành một phân phối phức tạp (chiếc bình hình ngôi sao).

## 2. Bài toán của AI: Tại sao cần một "người thợ gốm" giỏi?

Trong lĩnh vực mô hình sinh, mục tiêu của chúng ta là dạy cho máy tính cách tạo ra dữ liệu mới (ví dụ: ảnh khuôn mặt, giọng nói, văn bản) giống hệt dữ liệu thật.

Các "người thợ gốm" (mô hình) trước đây có một vài vấn đề:
- **VAE (Variational Autoencoder):** Giống như một người thợ mới vào nghề. Anh ta có thể tạo ra những chiếc bình trông khá giống hình ngôi sao, nhưng chúng thường hơi "mờ" và không sắc nét. Anh ta cũng chỉ có thể ước lượng "độ khó" để tạo ra một chiếc bình chứ không tính chính xác được.
- **GAN (Generative Adversarial Network):** Giống như một nghệ sĩ tài năng nhưng tính khí thất thường. Anh ta có thể tạo ra những chiếc bình hình ngôi sao cực kỳ đẹp và sắc nét. Tuy nhiên, quá trình dạy anh ta rất khó khăn (training không ổn định). Tệ hơn, anh ta không thể cho bạn biết xác suất để tạo ra một chiếc bình cụ thể là bao nhiêu. Anh ta chỉ biết "vẽ" thôi.

Đây là lúc chúng ta cần một phương pháp tốt hơn.

## 3. Normalizing Flow: Người thợ gốm vừa khéo tay, vừa minh bạch

**Normalizing Flow (NF)** là một loại mô hình sinh giống như một người thợ gốm bậc thầy, kết hợp ưu điểm của cả hai:

1.  **Sinh mẫu chất lượng cao:** Giống như GAN, NF có thể tạo ra dữ liệu sắc nét và chân thực.
2.  **Tính toán xác suất chính xác (Exact Likelihood):** Đây là điểm ăn tiền! Không giống GAN, NF có thể cho bạn biết chính xác xác suất để một mẫu dữ liệu (một chiếc bình cụ thể) tồn tại trong phân phối mà nó đã học. Điều này cực kỳ hữu ích trong các ứng dụng khoa học cần sự đo lường chính xác.

Cái tên "Normalizing" (chuẩn hóa) đến từ việc mô hình học cách biến đổi phân phối dữ liệu phức tạp *trở về* một phân phối "chuẩn" (thường là phân phối Gaussian). Vì phép biến đổi này đảo ngược được, chúng ta cũng có thể đi theo chiều ngược lại: từ phân phối chuẩn sinh ra dữ liệu phức tạp.

## 4. Phép màu toán học: Làm sao để theo dõi sự biến đổi?

Khi người thợ gốm biến đổi khối đất, mật độ của đất sét ở các vùng khác nhau sẽ thay đổi. Vùng bị kéo giãn ra sẽ có mật độ thấp hơn, vùng bị nén lại sẽ có mật độ cao hơn. Toán học cũng cần một cách để theo dõi sự "co giãn" này của không gian xác suất.

### 4.1. Công thức Biến đổi Biến số (Change of Variables)

Hãy bắt đầu với một ví dụ 1D siêu đơn giản.
Giả sử chúng ta có một biến ngẫu nhiên $z$ tuân theo phân phối Gaussian chuẩn (hình chuông đối xứng quanh 0). Ta định nghĩa một biến mới $x$ bằng một phép biến đổi đơn giản: $x = 2z + 1$.

- **Phép biến đổi:** $f(z) = 2z + 1$
- **Phép biến đổi ngược:** $f^{-1}(x) = (x-1)/2$

Phân phối của $x$ sẽ trông như thế nào? Nó vẫn là hình chuông, nhưng đã bị "kéo giãn" ra gấp 2 lần và "dịch chuyển" sang phải 1 đơn vị. Vì nó bị kéo giãn, chiều cao của đường cong mật độ xác suất phải giảm đi một nửa để đảm bảo tổng diện tích dưới đường cong vẫn bằng 1.

Công thức tổng quát cho sự thay đổi mật độ này là:
$$
p_x(x) = p_z(f^{-1}(x)) \left| \frac{d f^{-1}}{dx} \right|
$$
Trong ví dụ trên, nó bằng $|1/2| = 1/2$.

### 4.2. "Phí co giãn": Vai trò của Định thức Jacobian

Trong không gian nhiều chiều (ví dụ: một bức ảnh), "phí co giãn" không còn là một con số đơn giản nữa. Nó trở thành **định thức (determinant)** của **ma trận Jacobian**.

> **Ma trận Jacobian** là một ma trận chứa tất cả các đạo hàm riêng của phép biến đổi. Nó cho biết một vùng không gian nhỏ bị co giãn, xoay, và biến dạng như thế nào.
> **Định thức của Jacobian** là một con số duy nhất cho biết thể tích của vùng không gian đó thay đổi bao nhiêu lần.

Công thức log-likelihood trong không gian nhiều chiều trở thành:
$$
\log p_x(x) = \log p_z(z) - \log \left| \det \frac{\partial f}{\partial z} \right|
$$
Chúng ta lấy logarit vì nó biến phép nhân thành phép cộng, giúp việc tính toán và tối ưu dễ dàng hơn nhiều, đặc biệt khi chúng ta ghép nhiều phép biến đổi lại với nhau.

## 5. Thách thức thực tế: Vấn đề của dữ liệu lớn

Việc tính định thức Jacobian cho một ma trận $d \times d$ (với $d$ là số chiều dữ liệu) có độ phức tạp tính toán là $O(d^3)$. Với một bức ảnh 64x64 pixel, $d = 4096$. Con số này là không tưởng!

Đây là lúc các kiến trúc thông minh ra đời. Các mô hình như **RealNVP** hay **Glow** thiết kế các phép biến đổi (gọi là *coupling layers*) cực kỳ khéo léo để ma trận Jacobian luôn có dạng tam giác. Nhờ đó, định thức của nó chỉ đơn giản là tích các phần tử trên đường chéo. Độ phức tạp giảm từ $O(d^3)$ xuống chỉ còn $O(d)$! Đây là một bước đột phá giúp Normalizing Flow trở nên thực tế.

## 6. Continuous Normalizing Flow: Từ rời rạc đến liên tục

Các mô hình như RealNVP thực hiện một chuỗi các phép biến đổi *rời rạc*. Hãy tưởng tượng nó như một cuốn sách lật (flipbook), mỗi trang là một bước biến đổi.

**Continuous Normalizing Flow (CNF)** đưa ý tưởng này lên một tầm cao mới:
> Thay vì các bước nhảy rời rạc, tại sao không mô tả sự biến đổi như một dòng chảy *liên tục* và mượt mà theo thời gian?

Hãy quay lại ví dụ người thợ gốm. Thay vì xem từng động tác riêng lẻ, CNF mô tả toàn bộ quá trình như một video mượt mà. Về mặt toán học, nó mô tả "vận tốc" thay đổi của mỗi điểm trong không gian tại mỗi thời điểm, thông qua một **Phương trình vi phân thông thường (Ordinary Differential Equation - ODE)**.

Mô hình học một trường vector (vector field) $f(z, t)$ để chỉ hướng cho các điểm di chuyển. Lợi ích lớn nhất của CNF là nó mang lại sự linh hoạt tối đa cho kiến trúc mạng, vì chúng ta không còn bị ràng buộc bởi các phép biến đổi phải dễ tính Jacobian nữa. Đây chính là nền tảng cho các mô hình hiện đại hơn như **Flow Matching**.

## 7. Tổng kết và bước tiếp theo

- **Flow** là một chuỗi các phép biến đổi có thể đảo ngược.
- **Normalizing Flow** là một mô hình sinh mạnh mẽ, vừa tạo ra mẫu chất lượng cao, vừa tính được xác suất chính xác.
- Chìa khóa toán học là **công thức biến đổi biến số** và **định thức Jacobian** để theo dõi sự thay đổi mật độ xác suất.
- Các kiến trúc thông minh như **coupling layers** giúp NF trở nên khả thi trên thực tế.
- **CNF** là một bước tiến hóa, mô tả dòng chảy như một quá trình liên tục bằng ODE.

Hy vọng qua bài viết này, bạn đã có một cái nhìn trực quan và dễ hiểu về Normalizing Flow. Trong các bài viết tiếp theo, chúng ta sẽ đi sâu hơn vào các kiến trúc cụ thể và cách chúng hoạt động.

---

**Bài viết tiếp theo:**
- [Flow Matching: Từ lý thuyết đến thực hành](/posts/2025/flow-matching-theory)
- [Real NVP & Glow: Các kiến trúc có thể đảo ngược](/posts/2025/realnvp-glow)
