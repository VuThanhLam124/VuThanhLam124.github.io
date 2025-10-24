---
title: "Flow Matching: Từ Likelihood đến Regression"
date: "2025-01-20"
category: "flow-based-models"
tags: ["flow-matching", "optimal-transport", "generative-models", "regression", "pytorch"]
excerpt: "Flow Matching - phương pháp cách mạng hóa cách huấn luyện flow-based models. Người thợ gốm học cách định hình không qua likelihood phức tạp, mà qua regression đơn giản. Từ Conditional Flow Matching đến Optimal Transport, kèm chứng minh và code PyTorch đầy đủ."
author: "ThanhLamDev"
readingTime: 30
featured: true
---

# Flow Matching: Từ Likelihood đến Regression

**Khi Người Thợ Gốm Tìm Ra Cách Học Đơn Giản Hơn**

Chào mừng trở lại với series Flow-based Models! Trong bài [Normalizing Flows & CNF](/posts/2025/normalizing-flows), chúng ta đã gặp người thợ gốm bậc thầy - người biến khối đất sét đơn giản thành tác phẩm nghệ thuật phức tạp qua **dòng chảy liên tục** $\frac{dz(t)}{dt} = v_t(z(t))$.

Nhưng cách học của người thợ trong CNF/FFJORD gặp phải vấn đề nghiêm trọng: **quá phức tạp và đắt đỏ**. Flow Matching ra đời để giải quyết vấn đề này bằng một insight thiên tài: **Thay vì học likelihood, hãy học vector field trực tiếp qua regression!**
