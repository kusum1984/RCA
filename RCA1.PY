RP0441

SELECT 
  -- MKPF (Document headers)
  CAST(mkpf.BLDAT AS DATE) AS `Document Date`,
  CAST(mkpf.BUDAT AS DATE) AS `Posting Date`,

  -- MSEG (Line items)
  CAST(mseg.DMBTR AS INT) AS `Amount In LC`,
  mseg.CHARG AS `Batch`,
  mseg.WAERS AS `Currency`,
  CAST(mseg.MATNR AS STRING) AS `Material`,
  CAST(mseg.MBLNR AS STRING) AS `Material Document`,
  CAST(mseg.BWART AS STRING) AS `Movement Type`,
  CAST(mseg.WERKS AS STRING) AS `Plant`,
  CAST(mseg.ERFMG AS INT) AS `Quantity In UOE`,
  CAST(mseg.GRUND AS INT) AS `Reason for Movement`,
  CAST(mseg.BESTQ AS STRING) AS `Stock Category`,
  CAST(mseg.LGORT AS INT) AS `Storage Location`,
  mseg.ERFME AS `Unity of Entry`,

  -- MCH1
  CAST(mch1.ERNAM AS STRING) AS `Created By User`,

  -- T157E
  t157e.GRTXT AS `Reason for Movement Desc`,

  -- MARA
  mara.MTART AS `Material Type`,
  mara.BRGEW AS `Business Unit`, -- Assumed; update if different

  -- MAKT (Material description - Language-dependent)
  makt.MAKTX AS `Material Description`,

  -- CALCULATED COLUMNS
  CASE 
    WHEN mseg.BWART IN ('702', '704', '708', '711', '713', '715', '717') 
    THEN -1 * CAST(mseg.DMBTR AS INT) 
    ELSE CAST(mseg.DMBTR AS INT) 
  END AS `Net Amount In LC`,

  CASE 
    WHEN mseg.BWART IN ('702', '704', '708', '711', '713', '715', '717') 
    THEN -1 * CAST(mseg.ERFMG AS INT) 
    ELSE CAST(mseg.ERFMG AS INT) 
  END AS `Net Quantity In UOE`

FROM MSEG mseg
LEFT JOIN MKPF mkpf ON mseg.MANDT = mkpf.MANDT AND mseg.MBLNR = mkpf.MBLNR AND mseg.MJAHR = mkpf.MJAHR
LEFT JOIN MCH1 mch1 ON mseg.MANDT = mch1.MANDT AND mseg.CHARG = mch1.CHARG AND mseg.MATNR = mch1.MATNR
LEFT JOIN T157E t157e ON mseg.MANDT = t157e.MANDT AND mseg.GRUND = t157e.GRUND
LEFT JOIN MARA mara ON mseg.MANDT = mara.MANDT AND mseg.MATNR = mara.MATNR
LEFT JOIN MAKT makt ON mseg.MANDT = makt.MANDT AND mseg.MATNR = makt.MATNR AND makt.SPRAS = 'E' -- English

WHERE mseg.WERKS = 'CA01'




______________________


SELECT 
  -- MKPF (Document headers)
  CAST(mkpf.BLDAT AS DATE) AS `Document Date`,
  CAST(mkpf.BUDAT AS DATE) AS `Posting Date`,

  -- MSEG (Line items)
  CAST(mseg.DMBTR AS INT) AS `Amount In LC`,
  mseg.CHARG AS `Batch`,
  mseg.WAERS AS `Currency`,
  CAST(mseg.MATNR AS STRING) AS `Material`,
  CAST(mseg.MBLNR AS STRING) AS `Material Document`,
  CAST(mseg.BWART AS STRING) AS `Movement Type`,
  CAST(mseg.WERKS AS STRING) AS `Plant`,
  CAST(mseg.ERFMG AS INT) AS `Quantity In UOE`,
  CAST(mseg.GRUND AS INT) AS `Reason for Movement`,
  CAST(mseg.BESTQ AS STRING) AS `Stock Category`,
  CAST(mseg.LGORT AS INT) AS `Storage Location`,
  mseg.ERFME AS `Unity of Entry`,

  -- MCH1 (Batch created by user)
  CAST(mch1.ERNAM AS STRING) AS `Created By User`,

  -- T157E (Movement reason description)
  t157e.GRTXT AS `Reason for Movement Desc`,

  -- MARA (Material master data)
  mara.MTART AS `Material Type`,
  mara.MAKTX AS `Material Description`,
  mara.BRGEW AS `Business Unit`, -- assuming BRGEW is used for business unit; adjust if different

  -- CALCULATED COLUMNS
  CASE 
    WHEN mseg.BWART IN ('702', '704', '708', '711', '713', '715', '717') 
    THEN -1 * CAST(mseg.DMBTR AS INT) 
    ELSE CAST(mseg.DMBTR AS INT) 
  END AS `Net Amount In LC`,

  CASE 
    WHEN mseg.BWART IN ('702', '704', '708', '711', '713', '715', '717') 
    THEN -1 * CAST(mseg.ERFMG AS INT) 
    ELSE CAST(mseg.ERFMG AS INT) 
  END AS `Net Quantity In UOE`

FROM MSEG mseg
LEFT JOIN MKPF mkpf ON mseg.MANDT = mkpf.MANDT AND mseg.MBLNR = mkpf.MBLNR AND mseg.MJAHR = mkpf.MJAHR
LEFT JOIN MCH1 mch1 ON mseg.MANDT = mch1.MANDT AND mseg.CHARG = mch1.CHARG AND mseg.MATNR = mch1.MATNR
LEFT JOIN T157E t157e ON mseg.MANDT = t157e.MANDT AND mseg.GRUND = t157e.GRUND
LEFT JOIN MARA mara ON mseg.MANDT = mara.MANDT AND mseg.MATNR = mara.MATNR

WHERE mseg.WERKS = 'CA01'




*********************
WITH ekpo_filtered AS (
  SELECT *
  FROM EKPO
  WHERE WERKS = 'CA01'
),
-- From VBFA (PO to Delivery via Document Flow)
vbfa_link AS (
  SELECT VBELV AS PO_NUMBER, POSNV AS PO_ITEM, VBELN AS DELIVERY_NUMBER
  FROM VBFA
  WHERE VBTYP_V = 'F' AND VBTYP_N = 'J'  -- PO to Inbound Delivery
),
-- From EKES (PO to Delivery via Shipping Notification)
ekes_link AS (
  SELECT EBELN AS PO_NUMBER, EBELP AS PO_ITEM, VBELN AS DELIVERY_NUMBER
  FROM EKES
  WHERE VBELN IS NOT NULL
),
-- Union both mapping sources
po_delivery_link AS (
  SELECT * FROM vbfa_link
  UNION
  SELECT * FROM ekes_link
)

SELECT 
  -- EKPO
  ekpo_filtered.EBELN AS `PO Number`,
  ekpo_filtered.EBELP AS `PO Item`,
  ekpo_filtered.MATNR AS `Material`,
  ekpo_filtered.WERKS AS `Plant`,

  -- EKKO
  TO_DATE(ekko.AEDAT, 'yyyyMMdd') AS `Delivery Created Date`,

  -- EKET
  eket.CHARG AS `Batch`,
  TO_DATE(eket.EINDT, 'yyyyMMdd') AS `Delivery Date`,
  eket.GLMNG AS `Delivery Quantity Unit`,

  -- LFA1
  lfa1.LIFNR AS `Vendor`,
  lfa1.NAME1 AS `Vendor Name`,

  -- ADRC
  adrc.COUNTRY AS `Vendor Country`,

  -- T001W
  t001w.NAME1 AS `Plant Name`,

  -- LIPS
  lips.VGBEL AS `Reference Doc`,
  lips.VGPOS AS `Reference Doc Item`,
  lips.LGMNG AS `Actual Delivery Quantity`,
  lips.POSNR AS `Delivery Line`,
  lips.VBELN AS `Delivery Number`,
  lips.WBSTA AS `Delivery Itm GI Status`,

  -- LIKP
  likp.BOLNR AS `Bill of Lading`,
  likp.LFART AS `Delivery Type`,

  -- Calculated Fields
  CASE 
    WHEN lips.LGMNG > 0 THEN CONCAT(lips.VBELN, lips.POSNR) 
    ELSE NULL 
  END AS `Delivery Lines GT0`,

  COUNT(CASE WHEN lips.LGMNG > 0 THEN 1 END) OVER (PARTITION BY ekpo_filtered.EBELN) AS `Count of Delivery Lines GT0`,

  DATEDIFF(CURRENT_DATE, TO_DATE(eket.EINDT, 'yyyyMMdd')) AS `DaysPastDue`,
  CURRENT_DATE() AS `Report Date`,
  CURRENT_TIMESTAMP() AS `Report Time`

FROM ekpo_filtered
LEFT JOIN EKET eket 
  ON ekpo_filtered.EBELN = eket.EBELN AND ekpo_filtered.EBELP = eket.EBELP
LEFT JOIN EKKO ekko 
  ON ekpo_filtered.EBELN = ekko.EBELN
LEFT JOIN LFA1 lfa1 
  ON ekko.LIFNR = lfa1.LIFNR
LEFT JOIN ADRC adrc 
  ON lfa1.ADRNR = adrc.ADDRNUMBER
LEFT JOIN T001W t001w 
  ON ekpo_filtered.WERKS = t001w.WERKS

-- JOIN to Delivery via PO → Delivery mapping table
LEFT JOIN po_delivery_link link 
  ON ekpo_filtered.EBELN = link.PO_NUMBER AND ekpo_filtered.EBELP = link.PO_ITEM
LEFT JOIN LIKP likp 
  ON link.DELIVERY_NUMBER = likp.VBELN
LEFT JOIN LIPS lips 
  ON likp.VBELN = lips.VBELN

-- Only delivered lines with status 'C'
WHERE lips.WBSTA = 'C'


**********************************************



WITH ekpo_filtered AS (
  SELECT *
  FROM EKPO
  WHERE WERKS = 'CA01'
),
-- From VBFA (PO to Delivery via Document Flow)
vbfa_link AS (
  SELECT VBELV AS PO_NUMBER, POSNV AS PO_ITEM, VBELN AS DELIVERY_NUMBER
  FROM VBFA
  WHERE VBTYP_V = 'F' AND VBTYP_N = 'J'  -- PO to Inbound Delivery
),
-- From EKES (PO to Delivery via Shipping Notification)
ekes_link AS (
  SELECT EBELN AS PO_NUMBER, EBELP AS PO_ITEM, VBELN AS DELIVERY_NUMBER
  FROM EKES
  WHERE VBELN IS NOT NULL
),
-- Union both mapping sources
po_delivery_link AS (
  SELECT * FROM vbfa_link
  UNION
  SELECT * FROM ekes_link
)

SELECT 
  -- EKPO
  ekpo_filtered.EBELN AS `PO Number`,
  ekpo_filtered.EBELP AS `PO Item`,
  ekpo_filtered.MATNR AS `Material`,
  ekpo_filtered.WERKS AS `Plant`,

  -- EKKO
  ekko.AEDAT AS `Delivery Created Date`,

  -- EKET
  eket.CHARG AS `Batch`,
  eket.EINDT AS `Delivery Date`,
  eket.GLMNG AS `Delivery Quantity Unit`,

  -- LFA1
  lfa1.LIFNR AS `Vendor`,
  lfa1.NAME1 AS `Vendor Name`,

  -- ADRC
  adrc.COUNTRY AS `Vendor Country`,

  -- T001W
  t001w.NAME1 AS `Plant Name`,

  -- LIPS
  lips.VGBEL AS `Reference Doc`,
  lips.VGPOS AS `Reference Doc Item`,
  lips.LGMNG AS `Actual Delivery Quantity`,
  lips.POSNR AS `Delivery Line`,
  lips.VBELN AS `Delivery Number`,
  lips.WBSTA AS `Delivery Itm GI Status`,

  -- LIKP
  likp.BOLNR AS `Bill of Lading`,
  likp.LFART AS `Delivery Type`,

  -- Calculated Fields
  CASE 
    WHEN lips.LGMNG > 0 THEN CONCAT(lips.VBELN, lips.POSNR) 
    ELSE NULL 
  END AS `Delivery Lines GT0`,

  COUNT(CASE WHEN lips.LGMNG > 0 THEN 1 END) OVER (PARTITION BY ekpo_filtered.EBELN) AS `Count of Delivery Lines GT0`,

  DATEDIFF(CURRENT_DATE, eket.EINDT) AS `DaysPastDue`,
  CURRENT_DATE() AS `Report Date`,
  CURRENT_TIMESTAMP() AS `Report Time`

FROM ekpo_filtered
LEFT JOIN EKET eket ON ekpo_filtered.EBELN = eket.EBELN AND ekpo_filtered.EBELP = eket.EBELP
LEFT JOIN EKKO ekko ON ekpo_filtered.EBELN = ekko.EBELN
LEFT JOIN LFA1 lfa1 ON ekko.LIFNR = lfa1.LIFNR
LEFT JOIN ADRC adrc ON lfa1.ADRNR = adrc.ADDRNUMBER
LEFT JOIN T001W t001w ON ekpo_filtered.WERKS = t001w.WERKS

-- JOIN to Delivery via PO → Delivery mapping table
LEFT JOIN po_delivery_link link ON ekpo_filtered.EBELN = link.PO_NUMBER AND ekpo_filtered.EBELP = link.PO_ITEM
LEFT JOIN LIKP likp ON link.DELIVERY_NUMBER = likp.VBELN
LEFT JOIN LIPS lips ON likp.VBELN = lips.VBELN

-- Only delivered lines with status 'C'
WHERE lips.WBSTA = 'C'
