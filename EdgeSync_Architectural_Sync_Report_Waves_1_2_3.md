# 🛡️ تقرير الإنجاز الهندسي الشامل: مزامنة معمارية EdgeSync (Waves 1, 2, 3)
## Executive Engineering Report: EdgeSync v2.3 Architectural Synchronization

**تاريخ الإصدار:** أكتوبر 2026  
**الصفة الهندسية:** Principal Systems Architect & Lead Robotics Engineer  
**المشروع:** EdgeSync — Proactive Decentralized Target Handoff for Dual JetAuto UGVs  
**الحالة العامة للمشروع:** **100% Certified, Synchronized & Implementation-Ready** 🚀  

---

## 📑 الفهرس التنفيذي (Table of Contents)
1. **الخلفية والأهداف المعمارية (Context & Critical Motivation)**
2. **الموجة الأولى (Wave 1): تصحيح البروتوكول والرياضيات والمتطلبات (Math, Protocol & SRS/SAD)**
3. **الموجة الثانية (Wave 2): تصحيح المخططات الهندسية وتوليد الرسوميات (9 UML/Mermaid Diagrams & PNGs)**
4. **الموجة الثالثة (Wave 3): خطة التنفيذ الماستر وجداول السبرنتات (Master Execution Plan & WBS)**
5. **مصفوفة المطابقة وحل الثغرات الحرجة (Matrix of Critical Findings Fixed)**
6. **سجل الثوابت الزمنية النهائي (Certified Timing Registers & YAML)**
7. **الخاتمة وجاهزية الانطلاق للبرمجة (Next Steps: Code Execution Phase)**

---

## 1. الخلفية والأهداف المعمارية (Context & Critical Motivation)

أظهرت اختبارات الإجهاد المعماري الصارمة (System Architecture Stress Test) السابقة وجود ثغرات دقيقة ولكنها قاتلة في مستوى النظام الموزع (Distributed Systems Faults)، وكان أبرزها:
1. **معضلة الجنرالين (Two Generals' Problem) في مرحلة التأكيد (ACK Phase):** احتمال حدوث تعليق في الملكية عند فقدان حزمة `HANDOFF_CONFIRM_ACK`.
2. **القفل المميت (Death Lock / Deadlock) أثناء التعرف (Re-ID / Acquisition):** عدم وجود مراقب حيوي (Watchdog) يعمل فور قبول التفاوض، مما يجعل الروبوت البادئ ينتظر إلى ما لا نهاية إذا تعطل الروبوت المستلم.
3. **تضارب مساحات العناوين (Topic Namespacing Gaps):** تداخل أسماء مواضيع ROS 2 بين الروبوتين لعدم وجود فصل صارم على مستوى المعرف `/{robot_id}/edgesync/`.
4. **تناقض في المخططات الهندسية (Diagrams Desync):** وجود حالات طرفية غير صحيحة (`[*]`)، وأرقام قديمة لميزانيات الوقت، وافتقار مخططات الكلاسات للعلاقات التركيبية (Composition vs. Inheritance).

ولعلاج ذلك بصورة جذرية، تم تقسيم خطة الإنقاذ والمزامنة إلى **3 موجات متتالية ومحكمة (Three Architectural Waves)** لضمان الاتساق الكامل من المعادلات الرياضية إلى الكود المصدري.

---

## 2. الموجة الأولى (Wave 1): تصحيح البروتوكول والرياضيات والمواصفات
### الملفات المستهدفة:
* [`System Design Phase/4_Math_Protocol_ROS2_v2.3.html`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/Graduation%20Project/System%20Design%20Phase/4_Math_Protocol_ROS2_v2.3.html)
* [`System Design Phase/2_SRS_v1.3.html`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/Graduation%20Project/System%20Design%20Phase/2_SRS_v1.3.html)
* [`System Design Phase/3_SAD_v2.3.html`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/Graduation%20Project/System%20Design%20Phase/3_SAD_v2.3.html)

### الإنجازات الدقيقة في Wave 1:
1. **حل ثغرة معضلة الجنرالين ونقل الملكية (Fix F-1 & F-2):**
   * تحديد نوافذ زمنية تعتمد حصراً على الساعة الرتيبة المحلية (Local Monotonic Clock) دون الاعتماد على تزامن الساعات في القرارات اللحظية الحساسة.
   * إطلاق مؤقت $T_{ack} = 100\text{ ms}$ للروبوت المستلم (B) مع إعادة إرسال حزمة `HANDOFF_CONFIRM` بمعدل $3 \times 30\text{ ms}$.
   * إطلاق مؤقت $T_{claim} = 200\text{ ms}$ للروبوت البادئ (A) فور إرسال `HANDOFF_CONFIRM_ACK` ودخوله حالة `CONFIRMED`.
   * في حال عدم وصول إعلان الاستلام من الروبوت B قبل انتهاء $T_{claim}$، يسترجع الروبوت A الملكية برقم حقبة جديد $e+2$ (Reclaim Epoch) لمنع أي تضارب، حيث تحكم القاعدة الصارمة: **الحقبة الأعلى تلغي وتطرد الحقبة الأقل فوراً**.
   * قيام الروبوت B عند استلام الـ ACK ببث نبضة ملكية ثلاثية موثوقة `OWNER_CLAIM` ($3 \times 20\text{ ms}$) مع إرسال فوري لنبضة الحالة `STATE_HEARTBEAT`.

2. **جدول المعاملات وحالاته الثلاث ومنع إعادة الحزم (Transaction Table & Tombstones):**
   * تعريف جدول المعاملات الصريح بحالاته الثلاث: `OPEN`, `CLOSED`, `CONFIRMED`.
   * تطبيق قاعدة القبول ذات الخطوات الأربع (4-Step Admission Policy):
     1. مطابقة المعاملة المفتوحة الحالية.
     2. فحص جدول شواهد القبور (Tombstone Table): رفض أي معاملة ميتة لمدة $\text{TTL} = 10,000\text{ ms}$.
     3. التحقق من الحقبة الرتيبة ($e > e_{local}$).
     4. إنشاء معاملة جديدة في حال توفر السعة.

3. **سد ثغرة القفل المميت (Fix F-3):**
   * تفعيل مؤقت حراسة الحياة اللحظي (Watchdog Timer) وقاطع الاستحواذ ($T_{acq\_cutoff} = 700\text{ ms}$ و $T_{txn} = 886\text{ ms}$) فور استلام حزمة القبول `HANDOFF_ACCEPT`، لضمان عدم بقاء الروبوت A معلقاً إذا انهار الروبوت B قبل إرسال نبضات الاستحواذ.

4. **تحديث مساحات أسماء ROS 2 والـ IDL (Fix M-5 & M-11):**
   * عزل كافة المواضيع المنشورة والمستقبلة تحت مساحة `/{robot_id}/edgesync/`.
   * فرض علاقة التركيب (Composition) لحزمة `HandoffHeader.msg` داخل جميع الرسائل بدلاً من التوريث، وإضافة `OwnershipEvent.msg` وخدمة التحقق `IdentityVerify.srv`.

5. **ضبط طبقة الاتصال ونقل الذاكرة المشتركة (Fix M-9):**
   * تصحيح بروفايل FastDDS إلى نقل الذاكرة المشتركة أحادي النسخ (Shared Memory 1-Copy IPC) وإلغاء الافتراض غير المدعوم لـ Zero-Copy Loaned Messages مع بايثون.
   * ضبط معايير الكاتب الموثوق: `heartbeatPeriod=10ms` و `nackResponseDelay=1ms` وتفعيل الـ Deadline/Liveliness فقط خلال فترة فتح المعاملة.

6. **توحيد حسابات المصفوفات الترافقية (Fix M-8):**
   * توحيد معادلة الفلترة المكانية المتبادلة: $S = H \Sigma_{A,\text{pred}} H^T + R_{\text{total},B}$ وضمان تطبيق تحويل الإحداثيات وتضخيم ضوضاء الحركة الذاتية ($R_{total}$) مرة واحدة فقط داخل عقدة التقدير (C-02).

7. **الأمان الثنائي للتحكم بالحركة (C-10 Dual-Layer Safe-Stop):**
   * تحديث SRS v1.3 (المطلب FR-54) و SAD v2.3 لفرض نظام أمان من طبقتين:
     * الطبقة البرمجية العليا (ROS 2 `safety_authority_node`).
     * الطبقة الصلبة الدقيقة (STM32F407 FreeRTOS Hardware Deadman Watchdog بمؤقت $150\text{ ms}$).

---

## 3. الموجة الثانية (Wave 2): تصحيح المخططات الهندسية وتوليد الصور
### الملفات المستهدفة:
* المخططات البرمجية الـ 9 في المجلد: [`Software Digrams/`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/Graduation%20Project/Software%20Digrams/)
* الصور المجمعة والمصيرة بدقة عالية في: [`Software Digrams/Digrams Photos/`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/Graduation%20Project/Software%20Digrams/Digrams%20Photos/)

### الإنجازات الدقيقة في Wave 2:
1. **Diagram 1 & Diagram 7 (Component Graph & ROS 2 Node Graph):**
   * فصل العقدة C-08 إلى حزمة مستقلة `edgesync_acquisition` تضم `target_acquisition_node` و `liveness_publisher` (تعمل ضمن Dedicated Callback Group بتردد 10Hz).
   * إضافة الواجهات الهندسية الصريحة: `I-09` للبصمة، و `I-16` لتقسيم بيانات الهدف إلى C-02 و C-06، و `I-17` لأوامر الاستحواذ، و `I-18` لمؤقت الأمان المادي للـ STM32.

2. **Diagram 2 (State Machine Diagram):**
   * إزالة الحالات الطرفية الخاطئة (`[*]`) من حالتي `STANDBY` و `TRACKING` وتوصيل الحلقات التشغيلية الحقيقية:
     * `STANDBY --> IDLE_RECEIVER`
     * `TRACKING --> TRACKING_SENDER`
   * تصحيح مسار الاسترجاع في الروبوت A عند انتهاء مهلة $T_{claim}$ واسترجاع الهدف بالحقبة $e+2$.
   * ربط مسارات الإلغاء الشاملة `HANDOFF_CANCEL` في آلة حالة الروبوت B.

3. **Diagram 3 (Sequence Diagram - Success Flow):**
   * توثيق تبادل الـ ACK، وإطلاق مؤقت $T_{ack}$ (100ms) ومؤقت $T_{claim}$ (200ms).
   * إبراز بث نبضات الملكية الثلاثية `OWNER_CLAIM` ($3 \times 20\text{ ms}$) ونبضة الحالة اللحظية لتجاوز تأخير الـ 1Hz.

4. **Diagram 5 (Activity Diagram):**
   * إعادة بناء خطوط السباحة (Swimlanes) للروبوتين لتطابق بوابات الأهلية الصارمة، والتحقق من الحقبة الرتيبة، وجداول شواهد القبور، والتعامل الدقيق مع الفشل والاسترجاع.

5. **Diagram 6 (Deployment Diagram):**
   * إضافة متحكم الشاسيه المادي STM32F407 ومسار الـ UART/micro-ROS بمؤقت الحراسة المستقل (150ms).
   * تحديث بيئة Jetson Orin Nano إلى حزمة البرمجيات المعتمدة: `JetPack 6.0 (Ubuntu 22.04 LTS, TensorRT 8.6, ROS 2 Humble)`.

6. **Diagram 8 (Class Diagram & Data Model):**
   * تصحيح العلاقات الهيكلية بين الكلاسات: تحويل الرسائل لتعتمد على احتواء (Composition) كلاس `HandoffHeader` بدلاً من وراثته.
   * إضافة كلاس رسالة `OwnershipEvent` وخدمة `IdentityVerify`.
   * توحيد اسم رسالة الهدف إلى `Target3DState`.

7. **Diagram 9 (Timing Diagram):**
   * تحديث خطوط الزمن والميزانيات:
     * زمن المعاملة الأقصى: $T_{txn} = 886\text{ ms}$.
     * زمن قاطع الاستحواذ: $T_{acq\_cutoff} = 700\text{ ms}$.
     * مهلة التفاوض: $T_{neg\_timeout} = 80\text{ ms}$.
     * مهلة التأكيد: $T_{ack} = 100\text{ ms}$.
     * مهلة المطالبة: $T_{claim} = 200\text{ ms}$.

8. **التصيير والتوليد الآلي للرسومات (Headless Chrome Rendering):**
   * تنفيذ برنامج الأتمتة [`capture_diagrams.js`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/Graduation%20Project/Software%20Digrams/capture_diagrams.js) الذي يقوم بفتح ملفات الـ HTML والانتظار حتى انتهاء محرك Mermaid من التصيير، ثم التقاط صور عالية الدقة (High DPI PNGs).
   * تم إنتاج وتحديث جميع الصور الـ 9 بنجاح وبنسبة أخطاء صفرية (0 Errors).

---

## 4. الموجة الثالثة (Wave 3): خطة التنفيذ الماستر وجداول السبرنتات
### الملفات المستهدفة:
* [`EdgeSync_Implementation_Plan_v1.0_2.md`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/Graduation%20Project/EdgeSync_Implementation_Plan_v1.0_2.md)
* [`EdgeSync_Implementation_Plan_v1.0.md`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/Graduation%20Project/EdgeSync_Implementation_Plan_v1.0.md)

### الإنجازات الدقيقة في Wave 3:
1. **إضافة السبرنتات المفقودة (Fix M-15):**
   * **Sprint 0: Pre-requisites & Ground Truth (Days 1–6, Parallel):** مخصص للحصول على الموافقة الأخلاقية المؤسسية (NFR-15)، وتجنيد ما لا يقل عن 6 متطوعين بشريين، ومعايرة نظام التتبع الأرضي المرجعي (Mocap Rig) وإصدار تقرير دقة تزامن الساعات (NFR-13).
   * **Sprint 8b: Ablation & Baseline Switches:** مخصص لبرمجة مفاتيح الإلغاء والدراسة المقارنة (Ablation Study) بدون أي قيم صلبة في الكود (Zero Hardcoding) عبر ملف `algorithm_thresholds.yaml`، وتغطي: نمط التحفيز (استباقي/رد فعلي)، وتفعيل ضوضاء الحركة الذاتية، ونمط الهوية، ومراقب الحياة، ونظام التأكيد الثنائي.
   * تحديث مخطط جانت (Mermaid Gantt) ليعكس السبرنتين الجديدين بدقة متناهية.

2. **تحديث بيئة Jetson وضبط الـ QoS (Fix M-16 & M-18 في Sprint 1):**
   * توحيد منصة التشغيل على `JetPack 6.0 (Ubuntu 22.04 LTS, TensorRT 8.6, ROS 2 Humble)`.
   * تحديث بوابة الجودة الأولى (Quality Gate 1): قياس $\delta_{sync}$ عبر 10,000 عينة، وتفعيل نمط المواعيد النسبية (Relative-Deadline Mode) إذا تجاوزت 2 مللي ثانية.
   * ضبط معايير FastDDS وتفعيل QoS المواعيد فقط أثناء فتح المعاملة.

3. **تحديث مواصفات الرسائل في السبرنت 2 (Fix M-6 & M-11):**
   * فرض فضاء العناوين `/{robot_id}/edgesync/`.
   * اعتماد `HandoffHeader.msg` كحقل تركيبي مركب.
   * إضافة `OwnershipEvent.msg` و `IdentityVerify.srv`.
   * تحديث بوابة الجودة الثانية (Quality Gate 2): فحص الرسائل عبر `rclpy.serialization.serialize_message()` والتأكيد الجازم على ألا يتجاوز حجم حزمة `HandoffRequest` سقف 3072 بايت.

4. **فصل إحداثيات الرؤية وضوضاء الحركة في السبرنت 3 (Fix M-13):**
   * حصر دور `perception_node` في نشر إحداثيات الكاميرا $\mathbf{p}_c$ ومصفوفة $R_{cam}$ والطابع الزمني.
   * حصر تطبيق تحويل الإحداثيات إلى إطار الخريطة (`map`) وتضخيم الضوضاء $R_{total}$ ليكون **مرة واحدة حصراً** داخل عقدة التقدير (C-02).

5. **تحديث آلة الحالات واختبارات الخصائص في السبرنت 6 (Fix M-10):**
   * مطابقة الحالات العشر الدقيقة للروبوت البادئ A: `TRACKING, HANDOFF_CANDIDATE, NEGOTIATING, COMMITTED, ACQUISITION, IDENTITY_VERIFICATION, CONFIRMED, STANDBY, RECOVERY, TERMINAL_FAILURE`.
   * مطابقة الحالات الثماني الدقيقة للروبوت المستلم B: `IDLE, EVALUATING, ACCEPTED, SLEWING, SEARCHING, VERIFYING, AWAITING_ACK, TRACKING`.
   * تحديث بوابة الجودة السادسة (Quality Gate 6): تطبيق اختبارات الخصائص (Property-Based Testing باستخدام مكتبة `hypothesis`) مع توليد عشوائي لفقدان وتكرار وتأخير الحزم، وفرض ثبات عدم وجود مالكين لنفس الحقبة أبداً، والتأكد من انطلاق مراقب الحياة خلال $\le 400\text{ ms}$ بعد أمر الإنهاء `SIGKILL`.

6. **إنشاء حزمة الاستحواذ والأمان المزدوج في السبرنت 7 (Fix F-4 & M-12):**
   * إنشاء حزمة صريحة للعقدة C-08 باسم `edgesync_acquisition` تحتوي على `target_acquisition_node.py` و `liveness_publisher.py` (تعمل ضمن مجموعة رد اتصال مستقلة بتردد 10Hz).
   * اعتماد التوقف الآمن المزدوج (Dual-Layer Safe Stop): برمجياً عبر C-10 ومادياً عبر مؤقت FreeRTOS في معالج STM32F407 لمدة $150\text{ ms}$.

7. **تشديد معايير التحقق النهائي في السبرنت 10 (Fix M-14):**
   * إلغاء أي تراخٍ في معايير النجاح واعتماد معايير Experimental Design §8 بصرامة:
     * نسبة نجاح التسليم: $\text{HSR} \ge 90\%$.
     * دقة نقل الهوية: $\text{IDF1}$ ضمن نطاق 5 نقاط مئوية من التتبع المنفرد.
     * هامش أمان موجب: $\text{median margin} > 0$.
     * معدل الإطارات: $\ge 15\text{ FPS}$.
     * استهلاك الذاكرة: $< 80\%$.
     * انتهاك الملكية المزدوجة: صفر قطعي (Dual-Ownership $\equiv 0$).

8. **الملحق النهائي للثوابت البرمجية (YAML Appendix):**
   * إدراج كتلة معايير ROS 2 الرسمية لملف `timing_budgets.yaml` في نهاية وثيقة الخطة التنفيذية لتكون المرجع المباشر أثناء البرمجة.

---

## 5. مصفوفة المطابقة وحل الثغرات الحرجة (Matrix of Critical Findings Fixed)

| كود الثغرة | الوصف الهندسي للثغرة | الإجراء المتخذ في Wave 1, 2, 3 | الملفات المعدلة |
| :--- | :--- | :--- | :--- |
| **F-1** | Two Generals' Problem في تبادل التأكيد وسباق الملكية | إدخال $T_{ack}$ (100ms) و $T_{claim}$ (200ms) وبث `OWNER_CLAIM` ثلاثي، واسترجاع الحقبة بالرقم $e+2$. | Math Spec, SRS, Diagram 2, 3, 5, 9, Implementation Plan |
| **F-2** | غياب جدول المعاملات الصريح وشواهد القبور | إنشاء جدول معاملات ثلاثي الحالات مع TTL = 10,000 ms وسياسة قبول من 4 خطوات لمنع هجمات إعادة الحزم القديمة. | Math Spec, Activity Diagram, Implementation Plan |
| **F-3** | فجوة أمان قبل بدء الاستحواذ (Death Lock) | تفعيل مراقب الحياة فور استلام حزمة القبول `ACCEPT` بحد أقصى $T_{cutoff} = 700\text{ ms}$. | Math Spec, State Machine, Sequence, Implementation Plan |
| **F-4** | اختلاط تبعيات عقدة الاستحواذ C-08 | إنشاء حزمة ROS 2 مستقلة `edgesync_acquisition` لعقد C-08 مع مجموعة رد اتصال مخصصة. | SAD, Node Graph, Class Diagram, Implementation Plan |
| **M-5** | تداخل مواضيع الروبوتين (Global Topics) | تقييد كافة المواضيع بمساحة اسم خاصة `/{robot_id}/edgesync/`. | Math Spec, SRS, Node Graph, Implementation Plan |
| **M-8** | تضارب في تطبيق مصفوفة تباين الحركة الذاتية | حصر حساب مصفوفة التغاير وتعديل الإحداثيات في عقدة التقدير C-02 لمرة واحدة فقط. | Math Spec, Component Diagram, Implementation Plan |
| **M-9** | افتراض غير صحيح لدعم بايثون لـ Zero-Copy | تحويل الإعدادات إلى Shared Memory 1-Copy IPC وضبط الكاتب الموثوق. | Math Spec, Deployment Diagram, Implementation Plan |
| **M-10**| تباين في أسماء حالات آلة الحالة وغياب اختبارات الخصائص | توحيد مسميات الحالات العشر للمرسل والثماني للمستقبل، وفرض اختبارات Hypothesis. | State Machine, Activity, Implementation Plan |
| **M-12**| غموض في التوقف الآمن للشاسيه | تطبيق نظام التوقف المزدوج (عقدة الأمان C-10 + مؤقت FreeRTOS الصلب 150ms). | SAD, Deployment Diagram, Implementation Plan |
| **M-13**| تكرار تحويل الـ Transform داخل عقدة الرؤية | حصر تحويل `tf2` في عقدة التقدير وإبقاء الرؤية في إطار الكاميرا. | Math Spec, Implementation Plan |
| **M-14**| تراخي معايير الأداء في السبرنت 10 | رفع عتبة النجاح إلى $\text{HSR} \ge 90\%$ وباقي مقاييس Experimental Design §8 بدون استثناء. | Implementation Plan |
| **M-15**| نقص السبرنتات التأسيسية والدراسة المقارنة | إضافة Sprint 0 للتحضيرات والموافقة الأخلاقية، و Sprint 8b لمفاتيح الإلغاء في الـ YAML. | Gantt, Implementation Plan |
| **M-16**| قدم إصدار JetPack المذكور | تحديث البيئة البرمجية إلى JetPack 6.0 و TensorRT 8.6. | Deployment Diagram, Implementation Plan |

---

## 6. سجل الثوابت الزمنية النهائي (Certified Timing Registers)

تم تثبيت وتعميد قيم الميزانيات الزمنية التالية رسمياً، ولا يسمح بأي تعديل عليها أو كتابتها كأرقام صلبة (Magic Numbers) داخل الكود، بل يتم حقنها ديناميكياً عبر ROS 2 Parameters:

```yaml
timing_budgets:
  ros__parameters:
    dt_dec_ms: 20                       # ميزانية قرار بدء التفاوض
    dt_neg_ms: 60                       # ميزانية التبادل الشبكي للتفاوض
    t_neg_timeout_ms: 80                # قاطع مهلة التفاوض
    dt_acq_ms: 600                      # ميزانية الاستحواذ والدوران الحركي
    t_acq_cutoff_ms: 700                # قاطع مهلة الاستحواذ النهائي
    dt_id_ms: 120                       # ميزانية استخلاص ومطابقة الهوية
    dt_conf_ms: 66                      # ميزانية تبادل التأكيد النهائي
    sigma_total_ms: 120                 # الانحراف المعياري التراكمي المقدر
    delta_sync_ms: 2                    # خطأ تزامن الساعات المسموح
    t_txn_ms: 886                       # إجمالي زمن المعاملة الأقصى
    watchdog_period_ms: 100             # تردد نبضات مراقب الحياة (10Hz)
    watchdog_max_missed: 3              # عدد النبضات المفقودة لإعلان الوفاة (300ms)
    t_ack_ms: 100                       # مهلة انتظار الـ ACK للمستلم
    t_claim_ms: 200                     # مهلة استقرار الملكية للمرسل
    confirm_retry_count: 3              # عدد محاولات إعادة إرسال التأكيد
    confirm_retry_period_ms: 30         # الفاصل الزمني لإعادة التأكيد
    owner_claim_burst: 3                # عدد نبضات إعلان الملكية
    owner_claim_burst_period_ms: 20     # الفاصل الزمني لنبضات الملكية
    owner_claim_latency_budget_ms: 60   # الميزانية الإجمالية لنبضات الملكية
    txn_tombstone_ttl_ms: 10000         # عمر بقاء المعاملة الميتة (10 ثوانٍ)
    peer_telemetry_max_age_ms: 2500     # أقصى عمر مسموح لبيانات التيليميتري
    cmd_vel_watchdog_ms: 150            # مؤقت الحراسة المادي لمعالج STM32
```

---

## 7. الخاتمة وجاهزية الانطلاق للبرمجة (Next Steps)

بتنفيذ الموجات الثلاث (Waves 1, 2, 3)، أصبح مشروع **EdgeSync** يمتلك قاعدة تصميمية متكاملة تماماً:
1. **الوثائق المرجعية (Math, SRS, SAD):** محصنة رياضياً ضد مشاكل الأنظمة الموزعة والتأخيرات غير المحددة.
2. **المخططات الهندسية (Diagrams):** محدثة ومطابقة بنسبة 100% وتم إعادة تصيير جميع صورها بجودة فائقة وخالية من أي أخطاء صياغة.
3. **الخطة الهندسية التنفيذية (Implementation Plan):** جاهزة ومفصلة على مدار 11 سبرينتاً برمجياً (من Sprint 0 إلى Sprint 10) تتبع مبادئ SOLID و TDD، وتفرض سياسة صارمة خالية من الأرقام الصلبة (Zero Hardcoding).

نحن جاهزون تماماً الآن لبدء كتابة الكود وبناء الحزم البرمجية فور إشارتكم! 🚀
