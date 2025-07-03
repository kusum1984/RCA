SELECT
  -- EKPO
  ekpo.EBELN AS `PO Number`,
  ekpo.EBELP AS `PO Item`,
  ekpo.MATNR AS `Material`,
  ekpo.WERKS AS `Plant`,
  t001w.NAME1 AS `Plant Name`,

  -- EKKO
  ekko.AEDAT AS `Delivery Created Date`,

  -- EKET
  eket.CHARG AS `Batch`,
  eket.EINDT AS `Delivery Date`,
  eket.GLMNG AS `Delivery Quantity unit`,

  -- LFA1 / ADRC
  lfa1.LIFNR AS `Vendor`,
  lfa1.NAME1 AS `Vendor Name`,
  adrc.COUNTRY AS `Vendor Country`,

  -- VBFA → LIKP / LIPS
  lips.VGBEL AS `Reference Doc`,
  lips.VGPOS AS `Reference Doc Item`,
  lips.LGMNG AS `Actual Delivery Quantity`,
  lips.POSNR AS `Delivery Line`,
  lips.VBELN AS `Delivery Number`,
  lips.WBSTA AS `Delivery Itm GI Status`,

  likp.BOLNR AS `Bill of Lading`,
  likp.LFART AS `Delivery Type`,

  -- Calculated Fields
  CASE 
    WHEN lips.LGMNG > 0 THEN CONCAT(lips.VBELN, lips.POSNR) 
    ELSE NULL 
  END AS `Delivery Lines GT0`,

  COUNT(
    CASE WHEN lips.LGMNG > 0 THEN 1 END
  ) OVER (PARTITION BY ekpo.EBELN, ekpo.EBELP) AS `Count of Delivery Lines GT0`,

  DATEDIFF(CURRENT_DATE(), eket.EINDT) AS `DaysPastDue`,
  CURRENT_DATE() AS `Report Date`,
  CURRENT_TIMESTAMP() AS `Report Time`

FROM EKPO ekpo

LEFT JOIN EKKO ekko 
  ON ekpo.EBELN = ekko.EBELN

LEFT JOIN EKET eket 
  ON ekpo.EBELN = eket.EBELN AND ekpo.EBELP = eket.EBELP

LEFT JOIN LFA1 lfa1 
  ON ekko.LIFNR = lfa1.LIFNR

LEFT JOIN ADRC adrc 
  ON lfa1.ADRNR = adrc.ADDRNUMBER

LEFT JOIN T001W t001w 
  ON ekpo.WERKS = t001w.WERKS

-- NEW JOIN: Connect EKPO to Delivery via VBFA
LEFT JOIN VBFA vbfa 
  ON ekpo.EBELN = vbfa.VBELV 
  AND ekpo.EBELP = vbfa.POSNV
  AND vbfa.VBTYP_N = 'J'  -- 'J' means delivery doc type

LEFT JOIN LIKP likp 
  ON vbfa.VBELN = likp.VBELN

LEFT JOIN LIPS lips 
  ON likp.VBELN = lips.VBELN AND vbfa.POSNN = lips.POSNR

-- Filter for plant and completed delivery item
WHERE ekpo.WERKS = 'CA01'
  AND lips.WBSTA = 'C'

*************************************


SELECT
  -- EKPO (Driving Table)
  ekpo.EBELN AS `PO Number`,
  ekpo.EBELP AS `PO Item`,
  ekpo.MATNR AS `Material`,
  ekpo.WERKS AS `Plant`,

  -- Calculated Filter (plant = 'CA01')
  t001w.NAME1 AS `Plant Name`,

  -- EKKO
  ekko.AEDAT AS `Delivery Created Date`,

  -- EKET
  eket.CHARG AS `Batch`,
  eket.EINDT AS `Delivery Date`,
  eket.GLMNG AS `Delivery Quantity unit`,

  -- LFA1 and ADRC
  lfa1.LIFNR AS `Vendor`,
  lfa1.NAME1 AS `Vendor Name`,
  adrc.COUNTRY AS `Vendor Country`,

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

  -- Calculated Columns
  CASE 
    WHEN lips.LGMNG > 0 THEN CONCAT(lips.VBELN, lips.POSNR) 
    ELSE NULL 
  END AS `Delivery Lines GT0`,

  COUNT(
    CASE 
      WHEN lips.LGMNG > 0 THEN 1 
      ELSE NULL 
    END
  ) OVER (PARTITION BY ekpo.EBELN, ekpo.EBELP) AS `Count of Delivery Lines GT0`,

  DATEDIFF(CURRENT_DATE(), eket.EINDT) AS `DaysPastDue`,
  CURRENT_DATE() AS `Report Date`,
  CURRENT_TIMESTAMP() AS `Report Time`

FROM EKPO ekpo

-- Only records from plant 'CA01'
WHERE ekpo.WERKS = 'CA01'

-- Join EKKO (PO Header)
LEFT JOIN EKKO ekko ON ekpo.EBELN = ekko.EBELN

-- Join EKET (Schedule lines)
LEFT JOIN EKET eket ON ekpo.EBELN = eket.EBELN AND ekpo.EBELP = eket.EBELP

-- Join LFA1 (Vendor master)
LEFT JOIN LFA1 lfa1 ON ekko.LIFNR = lfa1.LIFNR

-- Join ADRC (Vendor address)
LEFT JOIN ADRC adrc ON lfa1.ADRNR = adrc.ADDRNUMBER

-- Join T001W (Plant master)
LEFT JOIN T001W t001w ON ekpo.WERKS = t001w.WERKS

-- Join LIPS (Delivery Item)
LEFT JOIN LIPS lips ON ekpo.EBELN = lips.VGBEL AND ekpo.EBELP = lips.VGPOS AND lips.WBSTA = 'C'

-- Join LIKP (Delivery Header)
LEFT JOIN LIKP likp ON lips.VBELN = likp.VBELN


**********************************


SELECT
    -- Select the desired fields from each table, using aliases for clarity
    ekko.ebeln AS purchase_order_number,
    ekko.lifnr AS vendor_number_ekko,
    ekpo.ebelp AS item_number,
    eket.eindt AS delivery_date,
    eket.menge AS schedule_line_quantity,
    lfa1.lifnr AS vendor_number_lfa1,
    lfa1.name1 AS vendor_name,
    adrc.addrnumber AS address_number,
    adrc.street AS street_name,
    t001w.werks AS plant_number_t001w,
    t001w.name1 AS plant_name,
    likp.vbeln AS delivery_number,
    likp.wadat_ist AS goods_movement_date,
    lips.posnr AS delivery_item_number,
    lips.matnr AS material_number
FROM
    ekko  -- Purchasing document header
INNER JOIN
    ekpo ON ekko.ebeln = ekpo.ebeln -- Link EKKO to EKPO by Purchase Order Number
INNER JOIN
    eket ON ekko.ebeln = eket.ebeln AND ekpo.ebelp = eket.ebelp -- Link EKKO and EKPO to EKET by Purchase Order Number and Item Number
INNER JOIN
    lfa1 ON ekko.lifnr = lfa1.lifnr -- Link EKKO to LFA1 by Vendor Number
LEFT OUTER JOIN -- Use LEFT OUTER JOIN to include vendors without an address entry in ADRC
    adrc ON lfa1.adrnr = adrc.addrnumber -- Link LFA1 to ADRC by Address Number
LEFT OUTER JOIN -- Use LEFT OUTER JOIN as not all EKKO records will have a supplying plant
    t001w ON ekko.reswk = t001w.werks -- Link EKKO to T001W by Supplying Plant
LEFT OUTER JOIN -- Use LEFT OUTER JOIN as not all PO items will have corresponding deliveries
    lips ON ekpo.ebeln = lips.vgbeli AND ekpo.ebelp = lips.vgposi -- Link EKPO to LIPS by Purchase Order Number and Item Number of preceding document
LEFT OUTER JOIN -- Use LEFT OUTER JOIN as not all delivery items will have a delivery header
    likp ON lips.vbeln = likp.vbeln -- Link LIPS to LIKP by Delivery Number
WHERE
    -- Add your desired filtering criteria here
    -- For example:
    -- ekko.ebeln = 'YOUR_PO_NUMBER'
    -- OR
    -- likp.vbeln = 'YOUR_DELIVERY_NUMBER'
;

*****************************************
*************************************************



WITH filtered_ekpo AS (
    SELECT EBELN, EBELP, MATNR, WERKS
    FROM EKPO
    WHERE WERKS = 'CA01'
),
eket_selected AS (
    SELECT EBELN, EBELP, CHARG, EINDT, GLMNG
    FROM EKET
),
ekko_selected AS (
    SELECT EBELN, AEDAT, LIFNR
    FROM EKKO
),
lfa1_selected AS (
    SELECT LIFNR, NAME1 AS VENDOR_NAME, ADRNR
    FROM LFA1
),
adrc_selected AS (
    SELECT ADDRNUMBER, COUNTRY
    FROM ADRC
),
t001w_selected AS (
    SELECT WERKS, NAME1 AS PLANT_NAME
    FROM T001W
),
vbfa_filtered AS (
    SELECT VBELV, VBELN, POSNN
    FROM VBFA
    WHERE VBTYP_N = 'J'  -- 'J' = Delivery
),
likp_selected AS (
    SELECT VBELN AS LIKP_VBELN, BOLNR, LFART
    FROM LIKP
),
lips_selected AS (
    SELECT VBELN AS LIPS_VBELN, POSNR, MATNR, VGBEL, VGPOS, LGMNG, WBSTA
    FROM LIPS
)

SELECT 
    ekko.AEDAT AS PO_CREATED_DATE,
    ekpo.EBELN AS PO_NUMBER,
    ekpo.EBELP AS PO_ITEM,
    ekpo.MATNR AS MATERIAL,
    ekpo.WERKS AS PLANT,
    t001w.PLANT_NAME,

    eket.CHARG AS BATCH,
    eket.EINDT AS SCHEDULED_DATE,
    eket.GLMNG AS SCHEDULE_QTY,

    lfa1.LIFNR AS VENDOR_ID,
    lfa1.VENDOR_NAME,
    adrc.COUNTRY AS VENDOR_COUNTRY,

    lips.LIPS_VBELN AS DELIVERY_NUM,
    lips.POSNR AS DELIVERY_ITEM,
    lips.LGMNG AS DELIVERED_QTY,
    lips.WBSTA AS DELIVERY_STATUS,

    likp.BOLNR AS BILL_OF_LADING,
    likp.LFART AS DELIVERY_TYPE

FROM filtered_ekpo ekpo
LEFT JOIN eket_selected eket 
    ON ekpo.EBELN = eket.EBELN AND ekpo.EBELP = eket.EBELP

LEFT JOIN ekko_selected ekko 
    ON ekpo.EBELN = ekko.EBELN

LEFT JOIN lfa1_selected lfa1 
    ON ekko.LIFNR = lfa1.LIFNR

LEFT JOIN adrc_selected adrc 
    ON lfa1.ADRNR = adrc.ADDRNUMBER

LEFT JOIN t001w_selected t001w 
    ON ekpo.WERKS = t001w.WERKS

-- Link PO to delivery through VBFA
LEFT JOIN vbfa_filtered vbfa 
    ON ekpo.EBELN = vbfa.VBELV AND ekpo.EBELP = vbfa.POSNN

-- Join to LIKP (delivery header)
LEFT JOIN likp_selected likp 
    ON vbfa.VBELN = likp.LIKP_VBELN

-- Join to LIPS (delivery item)
LEFT JOIN lips_selected lips 
    ON vbfa.VBELN = lips.LIPS_VBELN AND vbfa.POSNN = lips.POSNR;


**********************************************





WITH filtered_ekpo AS (
    SELECT 
        EBELN, EBELP, MATNR, WERKS
    FROM EKPO
    WHERE WERKS = 'CA01'
),
eket_selected AS (
    SELECT EBELN, EBELP, CHARG, EINDT, GLMNG
    FROM EKET
),
ekko_selected AS (
    SELECT EBELN, AEDAT, LIFNR
    FROM EKKO
),
lfa1_selected AS (
    SELECT LIFNR, NAME1 AS VENDOR_NAME, ADRNR
    FROM LFA1
),
adrc_selected AS (
    SELECT ADDRNUMBER, COUNTRY
    FROM ADRC
),
t001w_selected AS (
    SELECT WERKS, NAME1 AS PLANT_NAME
    FROM T001W
),
lips_selected AS (
    SELECT VGBEL, VGPOS, LGMNG, POSNR, VBELN, WBSTA, MATNR
    FROM LIPS
),
likp_selected AS (
    SELECT VBELN, BOLNR, LFART
    FROM LIKP
)

SELECT 
    ekko.AEDAT AS PO_CREATED_DATE,
    ekpo.MATNR AS MATERIAL,
    ekpo.WERKS AS PLANT,
    t001w.PLANT_NAME,
    
    eket.CHARG AS BATCH,
    eket.EINDT AS SCHEDULED_DATE,
    eket.GLMNG AS SCHEDULE_QTY,

    lfa1.LIFNR AS VENDOR_ID,
    lfa1.VENDOR_NAME,
    adrc.COUNTRY AS VENDOR_COUNTRY,

    lips.VGBEL AS PO_NUMBER_LINK,
    lips.VGPOS AS PO_ITEM_LINK,
    lips.LGMNG AS DELIVERED_QTY,
    lips.POSNR AS DELIVERY_ITEM,
    lips.WBSTA AS DELIVERY_STATUS,
    lips.VBELN AS DELIVERY_NUM,

    likp.BOLNR AS BILL_OF_LADING,
    likp.LFART AS DELIVERY_TYPE

FROM filtered_ekpo ekpo
LEFT JOIN eket_selected eket ON ekpo.EBELN = eket.EBELN AND ekpo.EBELP = eket.EBELP
LEFT JOIN ekko_selected ekko ON ekpo.EBELN = ekko.EBELN
LEFT JOIN lfa1_selected lfa1 ON ekko.LIFNR = lfa1.LIFNR
LEFT JOIN adrc_selected adrc ON lfa1.ADRNR = adrc.ADDRNUMBER
LEFT JOIN t001w_selected t001w ON ekpo.WERKS = t001w.WERKS
LEFT JOIN lips_selected lips ON ekpo.EBELN = lips.VGBEL AND ekpo.EBELP = lips.VGPOS
LEFT JOIN likp_selected likp ON lips.VBELN = likp.VBELN;


********************************************



SELECT 
  LIPS.VBELN AS `Delivery Number`,
  LIPS.POSNR AS `Delivery Line`,
  LIPS.VGBEL AS `Reference Doc`,
  LIPS.VGPOS AS `Reference Doc Item`,
  LIPS.LGMNG AS `Actual Delivery Quantity`,
  LIPS.GBSTA AS `Delivery Item GI Status`,
  EKPO.EBELN AS `PO Number`,
  EKPO.EBELP AS `PO Item`,
  EKPO.MATNR AS `Material`,
  EKPO.WERKS AS `Plant`,
  T001W.NAME1 AS `Plant Name`,
  EKKO.AEDAT AS `Delivery Created Date`,
  LFA1.LIFNR AS `Vendor`,
  LFA1.NAME1 AS `Vendor Name`,
  ADRC.COUNTRY AS `Vendor Country`,
  EKET.CHARG AS `Batch`,
  EKET.EINDT AS `Delivery Date`,
  LIKP.BOLNR AS `Bill of Lading`,
  LIKP.LFART AS `Delivery Type`
FROM LIPS
LEFT JOIN EKPO 
  ON LIPS.VGBEL = EKPO.EBELN 
  AND LIPS.VGPOS = EKPO.EBELP
LEFT JOIN T001W 
  ON EKPO.WERKS = T001W.WERKS
LEFT JOIN EKKO 
  ON EKPO.EBELN = EKKO.EBELN
LEFT JOIN LFA1 
  ON EKKO.LIFNR = LFA1.LIFNR
LEFT JOIN ADRC 
  ON LFA1.ADRNR = ADRC.ADDRNUMBER
LEFT JOIN EKET 
  ON EKPO.EBELN = EKET.EBELN 
  AND EKPO.EBELP = EKET.EBELP
LEFT JOIN LIKP 
  ON LIPS.VBELN = LIKP.VBELN
WHERE 
  LIPS.GBSTA = 'C'  -- Delivery Item GI Status filter
  AND EKPO.WERKS = 'CA01'  -- Plant filter
LIMIT 2;

***********************


SELECT 
  lips_renamed.`Delivery Number`,
  lips_renamed.`Delivery Line`,
  lips_renamed.`Reference Doc`,
  lips_renamed.`Reference Doc Item`,
  lips_renamed.`Actual Delivery Quantity`,
  likp_renamed.`Bill of Lading`,
  likp_renamed.`Delivery Type`,
  lips_renamed.`Delivery Item GI Status`,
  filtered_ekpo.`Material`,
  filtered_ekpo.`Plant`,
  t001w_renamed.`Plant Name`,
  ekko_renamed.`Delivery Created Date`,
  lfa1_renamed.`Vendor`,
  lfa1_renamed.`Vendor Name`,
  adrc_renamed.`Vendor Country`,
  eket_renamed.`Batch`,
  eket_renamed.`Delivery Date`,
  eket_renamed.`Delivery Quantity unit`,
  CASE 
    WHEN lips_renamed.`Actual Delivery Quantity` > 0 
    THEN CONCAT(lips_renamed.`Delivery Number`, ' - ', lips_renamed.`Delivery Line`) 
  END AS `Delivery Lines GT0`,
  CURRENT_DATE() AS `Report Date`,
  DATE_FORMAT(CURRENT_TIMESTAMP(), 'HH:mm:ss') AS `Report Time`,
  DATEDIFF(CURRENT_DATE(), eket_renamed.`Delivery Date`) AS `DaysPastDue`,
  COUNT(CASE 
    WHEN lips_renamed.`Actual Delivery Quantity` > 0 
    THEN CONCAT(lips_renamed.`Delivery Number`, ' - ', lips_renamed.`Delivery Line`) END
  ) OVER (PARTITION BY lips_renamed.`Delivery Number`) AS `Count of Delivery Lines GT0`
FROM (
  SELECT 
    VBELN AS `Delivery Number`,
    POSNR AS `Delivery Line`,
    VGBEL AS `Reference Doc`,
    VGPOS AS `Reference Doc Item`,
    LGMNG AS `Actual Delivery Quantity`,
    GBSTA AS `Delivery Item GI Status`
  FROM LIPS
  WHERE GBSTA = 'C'
) AS lips_renamed
INNER JOIN (
  SELECT 
    EBELN AS `PO Number`,
    EBELP AS `PO Item`,
    MATNR AS `Material`,
    WERKS AS `Plant`
  FROM EKPO
  WHERE WERKS = 'CA01'
) AS filtered_ekpo 
  ON lips_renamed.`Reference Doc` = filtered_ekpo.`PO Number`
  AND lips_renamed.`Reference Doc Item` = filtered_ekpo.`PO Item`
LEFT JOIN (
  SELECT 
    VBELN,
    BOLNR AS `Bill of Lading`,
    LFART AS `Delivery Type`
  FROM LIKP
) AS likp_renamed 
  ON lips_renamed.`Delivery Number` = likp_renamed.VBELN
LEFT JOIN (
  SELECT 
    EBELN,
    AEDAT AS `Delivery Created Date`,
    LIFNR
  FROM EKKO
) AS ekko_renamed 
  ON filtered_ekpo.`PO Number` = ekko_renamed.EBELN
LEFT JOIN (
  SELECT 
    LIFNR AS `Vendor`,
    NAME1 AS `Vendor Name`,
    ADRNR
  FROM LFA1
) AS lfa1_renamed 
  ON ekko_renamed.LIFNR = lfa1_renamed.`Vendor`
LEFT JOIN (
  SELECT 
    ADDRNUMBER,
    COUNTRY AS `Vendor Country`
  FROM ADRC
) AS adrc_renamed 
  ON lfa1_renamed.ADRNR = adrc_renamed.ADDRNUMBER
LEFT JOIN (
  SELECT 
    EBELN,
    EBELP,
    CHARG AS `Batch`,
    EINDT AS `Delivery Date`,
    GLMNG AS `Delivery Quantity unit`
  FROM EKET
) AS eket_renamed 
  ON filtered_ekpo.`PO Number` = eket_renamed.EBELN 
  AND filtered_ekpo.`PO Item` = eket_renamed.EBELP
LEFT JOIN (
  SELECT 
    WERKS AS `Plant`,
    NAME1 AS `Plant Name`
  FROM T001W
) AS t001w_renamed 
  ON filtered_ekpo.`Plant` = t001w_renamed.`Plant`
LIMIT 2;

***************************************



WITH 
  filtered_ekpo AS (
    SELECT 
      EBELN AS `PO Number`,
      EBELP AS `PO Item`,
      MATNR AS `Material`,
      WERKS AS `Plant`
    FROM EKPO
    WHERE WERKS = 'CA01'
  ),
  adrc_renamed AS (
    SELECT 
      ADDRNUMBER,
      COUNTRY AS `Vendor Country`
    FROM ADRC
  ),
  eket_renamed AS (
    SELECT 
      EBELN,
      EBELP,
      CHARG AS `Batch`,
      EINDT AS `Delivery Date`,
      GLMNG AS `Delivery Quantity unit`
    FROM EKET
  ),
  ekko_renamed AS (
    SELECT 
      EBELN,
      AEDAT AS `Delivery Created Date`,
      LIFNR
    FROM EKKO
  ),
  lfa1_renamed AS (
    SELECT 
      LIFNR AS `Vendor`,
      NAME1 AS `Vendor Name`,
      ADRNR
    FROM LFA1
  ),
  t001w_renamed AS (
    SELECT 
      WERKS AS `Plant`,
      NAME1 AS `Plant Name`
    FROM T001W
  ),
  lips_renamed AS (
    SELECT 
      VBELN AS `Delivery Number`,
      POSNR AS `Delivery Line`,
      VGBEL AS `Reference Doc`,
      VGPOS AS `Reference Doc Item`,
      LGMNG AS `Actual Delivery Quantity`,
      GBSTA AS `Delivery Item GI Status`  -- Changed from VBUP.WBSTA to LIPS.GBSTA
    FROM LIPS
    WHERE GBSTA = 'C'  -- Filter applied directly in LIPS
  ),
  likp_renamed AS (
    SELECT 
      VBELN,
      BOLNR AS `Bill of Lading`,
      LFART AS `Delivery Type`
    FROM LIKP
  ),
  joined_data AS (
    SELECT 
      lips_renamed.`Delivery Number`,
      lips_renamed.`Delivery Line`,
      lips_renamed.`Reference Doc`,
      lips_renamed.`Reference Doc Item`,
      lips_renamed.`Actual Delivery Quantity`,
      likp_renamed.`Bill of Lading`,
      likp_renamed.`Delivery Type`,
      lips_renamed.`Delivery Item GI Status`,  -- Now sourced from LIPS
      filtered_ekpo.`Material`,
      filtered_ekpo.`Plant`,
      t001w_renamed.`Plant Name`,
      ekko_renamed.`Delivery Created Date`,
      lfa1_renamed.`Vendor`,
      lfa1_renamed.`Vendor Name`,
      adrc_renamed.`Vendor Country`,
      eket_renamed.`Batch`,
      eket_renamed.`Delivery Date`,
      eket_renamed.`Delivery Quantity unit`,
      CASE 
        WHEN lips_renamed.`Actual Delivery Quantity` > 0 
        THEN CONCAT(lips_renamed.`Delivery Number`, ' - ', lips_renamed.`Delivery Line`) 
      END AS `Delivery Lines GT0`,
      CURRENT_DATE() AS `Report Date`,
      DATE_FORMAT(CURRENT_TIMESTAMP(), 'HH:mm:ss') AS `Report Time`,
      DATEDIFF(CURRENT_DATE(), eket_renamed.`Delivery Date`) AS `DaysPastDue`
    FROM lips_renamed
    INNER JOIN likp_renamed 
      ON lips_renamed.`Delivery Number` = likp_renamed.VBELN
    INNER JOIN filtered_ekpo 
      ON lips_renamed.`Reference Doc` = filtered_ekpo.`PO Number` 
      AND lips_renamed.`Reference Doc Item` = filtered_ekpo.`PO Item`
    LEFT JOIN ekko_renamed 
      ON filtered_ekpo.`PO Number` = ekko_renamed.EBELN
    LEFT JOIN eket_renamed 
      ON filtered_ekpo.`PO Number` = eket_renamed.EBELN 
      AND filtered_ekpo.`PO Item` = eket_renamed.EBELP
    LEFT JOIN lfa1_renamed 
      ON ekko_renamed.LIFNR = lfa1_renamed.`Vendor`
    LEFT JOIN adrc_renamed 
      ON lfa1_renamed.ADRNR = adrc_renamed.ADDRNUMBER
    LEFT JOIN t001w_renamed 
      ON filtered_ekpo.`Plant` = t001w_renamed.`Plant`
  )

SELECT 
  *,
  COUNT(`Delivery Lines GT0`) OVER (
    PARTITION BY `Delivery Number`
  ) AS `Count of Delivery Lines GT0`
FROM joined_data
LIMIT 2;

-- Sample Output (2 simulated rows):
-- | Delivery Number | Delivery Line | Reference Doc | Reference Doc Item | Actual Delivery Quantity | Bill of Lading | Delivery Type | Delivery Item GI Status | Material | Plant | Plant Name | Delivery Created Date | Vendor | Vendor Name | Vendor Country | Batch | Delivery Date | Delivery Quantity unit | Delivery Lines GT0 | Report Date | Report Time | DaysPastDue | Count of Delivery Lines GT0 |
-- |-----------------|---------------|---------------|--------------------|--------------------------|----------------|---------------|-------------------------|----------|-------|------------|------------------------|--------|-------------|----------------|-------|---------------|-------------------------|--------------------|-------------|-------------|-------------|-----------------------------|
-- | 0080000001      | 000010        | 4500000001    | 00010              | 100.00                   | BOL00001       | LF            | C                       | MAT0001  | CA01  | Toronto    | 2023-05-15            | V0001  | Vendor Inc  | CA             | BATCH1| 2023-05-20    | 100.00                 | 0080000001 - 000010| 2023-10-01  | 14:30:00    | 134         | 1                           |
-- | 0080000002      | 000020        | 4500000002    | 00020              | 50.00                    | BOL00002       | LF            | C                       | MAT0002  | CA01  | Toronto    | 2023-05-16            | V0002  | Supplier Co | US             | BATCH2| 2023-05-21    | 50.00                  | 0080000002 - 000020| 2023-10-01  | 14:30:00    | 133         | 1                           |

_______________________________________
*********************************************************************************************************************





WITH 
  filtered_ekpo AS (
    SELECT 
      EBELN AS `PO Number`,
      EBELP AS `PO Item`,
      MATNR AS `Material`,
      WERKS AS `Plant`
    FROM EKPO
    WHERE WERKS = 'CA01'
  ),
  adrc_renamed AS (
    SELECT 
      ADDRNUMBER,
      COUNTRY AS `Vendor Country`
    FROM ADRC
  ),
  eket_renamed AS (
    SELECT 
      EBELN,
      EBELP,
      CHARG AS `Batch`,
      EINDT AS `Delivery Date`,
      GLMNG AS `Delivery Quantity unit`
    FROM EKET
  ),
  ekko_renamed AS (
    SELECT 
      EBELN,
      AEDAT AS `Delivery Created Date`,
      LIFNR
    FROM EKKO
  ),
  lfa1_renamed AS (
    SELECT 
      LIFNR AS `Vendor`,
      NAME1 AS `Vendor Name`,
      ADRNR
    FROM LFA1
  ),
  t001w_renamed AS (
    SELECT 
      WERKS AS `Plant`,
      NAME1 AS `Plant Name`
    FROM T001W
  ),
  lips_renamed AS (
    SELECT 
      VBELN AS `Delivery Number`,
      POSNR AS `Delivery Line`,
      VGBEL AS `Reference Doc`,
      VGPOS AS `Reference Doc Item`,
      LGMNG AS `Actual Delivery Quantity`,
      WBSTA AS `Delivery Item GI Status`
    FROM LIPS
    WHERE WBSTA = 'C'
  ),
  likp_renamed AS (
    SELECT 
      VBELN,
      BOLNR AS `Bill of Lading`,
      LFART AS `Delivery Type`
    FROM LIKP
  ),
  joined_data AS (
    SELECT 
      LIPS.`Delivery Number`,
      LIPS.`Delivery Line`,
      LIPS.`Reference Doc`,
      LIPS.`Reference Doc Item`,
      LIPS.`Actual Delivery Quantity`,
      LIPS.`Delivery Item GI Status`,
      LIKP.`Bill of Lading`,
      LIKP.`Delivery Type`,
      EKPO.`Material`,
      EKPO.`Plant`,
      T001W.`Plant Name`,
      EKKO.`Delivery Created Date`,
      LFA1.`Vendor`,
      LFA1.`Vendor Name`,
      ADRC.`Vendor Country`,
      EKET.`Batch`,
      EKET.`Delivery Date`,
      EKET.`Delivery Quantity unit`,

      -- Calculated columns
      CASE 
        WHEN LIPS.`Actual Delivery Quantity` > 0 
        THEN CONCAT(LIPS.`Delivery Number`, ' - ', LIPS.`Delivery Line`) 
      END AS `Delivery Lines GT0`,
      CURRENT_DATE() AS `Report Date`,
      DATE_FORMAT(CURRENT_TIMESTAMP(), 'HH:mm:ss') AS `Report Time`,
      DATEDIFF(CURRENT_DATE(), EKET.`Delivery Date`) AS `DaysPastDue`
      
    FROM lips_renamed LIPS
    LEFT JOIN LIKP LIKP 
      ON LIPS.`Delivery Number` = LIKP.VBELN
    INNER JOIN filtered_ekpo EKPO 
      ON LIPS.`Reference Doc` = EKPO.`PO Number` AND LIPS.`Reference Doc Item` = EKPO.`PO Item`
    LEFT JOIN ekko_renamed EKKO 
      ON EKPO.`PO Number` = EKKO.EBELN
    LEFT JOIN eket_renamed EKET 
      ON EKPO.`PO Number` = EKET.EBELN AND EKPO.`PO Item` = EKET.EBELP
    LEFT JOIN lfa1_renamed LFA1 
      ON EKKO.LIFNR = LFA1.`Vendor`
    LEFT JOIN adrc_renamed ADRC 
      ON LFA1.ADRNR = ADRC.ADDRNUMBER
    LEFT JOIN t001w_renamed T001W 
      ON EKPO.`Plant` = T001W.`Plant`
  )

SELECT 
  *,
  COUNT(`Delivery Lines GT0`) OVER (
    PARTITION BY `Delivery Number`
  ) AS `Count of Delivery Lines GT0`
FROM joined_data
LIMIT 2;

============================================================












WITH 
  filtered_ekpo AS (
    SELECT 
      EBELN AS `PO Number`,
      EBELP AS `PO Item`,
      MATNR AS `Material`,
      WERKS AS `Plant`
    FROM EKPO
    WHERE WERKS = 'CA01'
  ),
  adrc_renamed AS (
    SELECT 
      ADDRNUMBER,
      COUNTRY AS `Vendor Country`
    FROM ADRC
  ),
  eket_renamed AS (
    SELECT 
      EBELN,
      EBELP,
      CHARG AS `Batch`,
      EINDT AS `Delivery Date`,
      GLMNG AS `Delivery Quantity unit`
    FROM EKET
  ),
  ekko_renamed AS (
    SELECT 
      EBELN,
      AEDAT AS `Delivery Created Date`,
      LIFNR
    FROM EKKO
  ),
  lfa1_renamed AS (
    SELECT 
      LIFNR AS `Vendor`,
      NAME1 AS `Vendor Name`,
      ADRNR
    FROM LFA1
  ),
  t001w_renamed AS (
    SELECT 
      WERKS AS `Plant`,
      NAME1 AS `Plant Name`
    FROM T001W
  ),
  lips_renamed AS (
    SELECT 
      VBELN AS `Delivery Number`,
      POSNR AS `Delivery Line`,
      VGBEL AS `Reference Doc`,
      VGPOS AS `Reference Doc Item`,
      LGMNG AS `Actual Delivery Quantity`
    FROM LIPS
  ),
  likp_renamed AS (
    SELECT 
      VBELN,
      BOLNR AS `Bill of Lading`,
      LFART AS `Delivery Type`
    FROM LIKP
  ),
  filtered_vbup AS (
    SELECT 
      VBELN,
      POSNR,
      WBSTA AS `Delivery Itm GI Status`
    FROM VBUP
    WHERE WBSTA = 'C'
  ),
  joined_data AS (
    SELECT 
      lip.`Delivery Number`,
      lip.`Delivery Line`,
      lip.`Reference Doc`,
      lip.`Reference Doc Item`,
      lip.`Actual Delivery Quantity`,
      lik.`Bill of Lading`,
      lik.`Delivery Type`,
      vbu.`Delivery Itm GI Status`,
      ekp.`Material`,
      ekp.`Plant`,
      tw.`Plant Name`,
      ekk.`Delivery Created Date`,
      lfa.`Vendor`,
      lfa.`Vendor Name`,
      adr.`Vendor Country`,
      eke.`Batch`,
      eke.`Delivery Date`,
      eke.`Delivery Quantity unit`,
      
      -- Calculated columns
      CASE 
        WHEN lip.`Actual Delivery Quantity` > 0 
        THEN CONCAT(lip.`Delivery Number`, ' - ', lip.`Delivery Line`) 
      END AS `Delivery Lines GT0`,
      CURRENT_DATE() AS `Report Date`,
      DATE_FORMAT(CURRENT_TIMESTAMP(), 'HH:mm:ss') AS `Report Time`,
      DATEDIFF(CURRENT_DATE(), eke.`Delivery Date`) AS `DaysPastDue`
      
    FROM lips_renamed lip
    INNER JOIN filtered_vbup vbu 
      ON lip.`Delivery Number` = vbu.VBELN AND lip.`Delivery Line` = vbu.POSNR
    LEFT JOIN likp_renamed lik 
      ON lip.`Delivery Number` = lik.VBELN
    INNER JOIN filtered_ekpo ekp 
      ON lip.`Reference Doc` = ekp.`PO Number` AND lip.`Reference Doc Item` = ekp.`PO Item`
    LEFT JOIN ekko_renamed ekk 
      ON ekp.`PO Number` = ekk.EBELN
    LEFT JOIN eket_renamed eke 
      ON ekp.`PO Number` = eke.EBELN AND ekp.`PO Item` = eke.EBELP
    LEFT JOIN lfa1_renamed lfa 
      ON ekk.LIFNR = lfa.`Vendor`
    LEFT JOIN adrc_renamed adr 
      ON lfa.ADRNR = adr.ADDRNUMBER
    LEFT JOIN t001w_renamed tw 
      ON ekp.`Plant` = tw.`Plant`
  )

SELECT 
  *,
  COUNT(`Delivery Lines GT0`) OVER (
    PARTITION BY `Delivery Number`
  ) AS `Count of Delivery Lines GT0`
FROM joined_data
-- Optional: Add LIMIT for initial testing
LIMIT 100;


















++++++++++++++++++++++++++++++++++++++++++++
CREATE OR REPLACE TABLE delivery_report AS
WITH 
  filtered_ekpo AS (
    SELECT 
      EBELN AS `PO Number`,
      EBELP AS `PO Item`,
      MATNR AS `Material`,
      WERKS AS `Plant`
    FROM EKPO
    WHERE WERKS = 'CA01'
  ),
  adrc_renamed AS (
    SELECT 
      ADDRNUMBER,
      COUNTRY AS `Vendor Country`
    FROM ADRC
  ),
  eket_renamed AS (
    SELECT 
      EBELN,
      EBELP,
      CHARG AS `Batch`,
      EINDT AS `Delivery Date`,
      GLMNG AS `Delivery Quantity unit`
    FROM EKET
  ),
  ekko_renamed AS (
    SELECT 
      EBELN,
      AEDAT AS `Delivery Created Date`,
      LIFNR
    FROM EKKO
  ),
  lfa1_renamed AS (
    SELECT 
      LIFNR AS `Vendor`,
      NAME1 AS `Vendor Name`,
      ADRNR
    FROM LFA1
  ),
  t001w_renamed AS (
    SELECT 
      WERKS AS `Plant`,
      NAME1 AS `Plant Name`
    FROM T001W
  ),
  lips_renamed AS (
    SELECT 
      VBELN AS `Delivery Number`,
      POSNR AS `Delivery Line`,
      VGBEL AS `Reference Doc`,
      VGPOS AS `Reference Doc Item`,
      LGMNG AS `Actual Delivery Quantity`
    FROM LIPS
  ),
  likp_renamed AS (
    SELECT 
      VBELN,
      BOLNR AS `Bill of Lading`,
      LFART AS `Delivery Type`
    FROM LIKP
  ),
  filtered_vbup AS (
    SELECT 
      VBELN,
      POSNR,
      WBSTA AS `Delivery Itm GI Status`
    FROM VBUP
    WHERE WBSTA = 'C'
  ),
  joined_data AS (
    SELECT 
      lip.`Delivery Number`,
      lip.`Delivery Line`,
      lip.`Reference Doc`,
      lip.`Reference Doc Item`,
      lip.`Actual Delivery Quantity`,
      lik.`Bill of Lading`,
      lik.`Delivery Type`,
      vbu.`Delivery Itm GI Status`,
      ekp.`Material`,
      ekp.`Plant`,
      tw.`Plant Name`,
      ekk.`Delivery Created Date`,
      lfa.`Vendor`,
      lfa.`Vendor Name`,
      adr.`Vendor Country`,
      eke.`Batch`,
      eke.`Delivery Date`,
      eke.`Delivery Quantity unit`,
      
      -- Calculated columns
      CASE 
        WHEN lip.`Actual Delivery Quantity` > 0 
        THEN CONCAT(lip.`Delivery Number`, ' - ', lip.`Delivery Line`) 
      END AS `Delivery Lines GT0`,
      CURRENT_DATE() AS `Report Date`,
      DATE_FORMAT(CURRENT_TIMESTAMP(), 'HH:mm:ss') AS `Report Time`,
      DATEDIFF(CURRENT_DATE(), eke.`Delivery Date`) AS `DaysPastDue`
      
    FROM lips_renamed lip
    INNER JOIN filtered_vbup vbu 
      ON lip.`Delivery Number` = vbu.VBELN AND lip.`Delivery Line` = vbu.POSNR
    LEFT JOIN likp_renamed lik 
      ON lip.`Delivery Number` = lik.VBELN
    INNER JOIN filtered_ekpo ekp 
      ON lip.`Reference Doc` = ekp.`PO Number` AND lip.`Reference Doc Item` = ekp.`PO Item`
    LEFT JOIN ekko_renamed ekk 
      ON ekp.`PO Number` = ekk.EBELN
    LEFT JOIN eket_renamed eke 
      ON ekp.`PO Number` = eke.EBELN AND ekp.`PO Item` = eke.EBELP
    LEFT JOIN lfa1_renamed lfa 
      ON ekk.LIFNR = lfa.`Vendor`
    LEFT JOIN adrc_renamed adr 
      ON lfa.ADRNR = adr.ADDRNUMBER
    LEFT JOIN t001w_renamed tw 
      ON ekp.`Plant` = tw.`Plant`
  )

SELECT 
  *,
  COUNT(`Delivery Lines GT0`) OVER (PARTITION BY `Delivery Number`) AS `Count of Delivery Lines GT0`
FROM joined_data;










import os
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain import hub
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import numpy as np

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHROMA_DB_PATH = "./chroma_db"

# Initialize components
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model_name="gpt-4",
    temperature=0.3
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model="text-embedding-ada-002"
)

# Initialize Chroma DB
vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)
"""
Purpose:

Converts historical CAPA PDFs into searchable vectors

Creates a knowledge base for semantic search

Why Used:

ChromaDB: Lightweight, local vector store that preserves data privacy

Embeddings: Convert text to vectors to enable meaning-based search (not just keyword matching)

Persistent storage: Retains knowledge between sessions"""
# --------------------------
# Semantic Search Tool
# --------------------------

def semantic_search(query: str) -> str:
    """Perform semantic search with meaning-based queries"""
    docs = vector_db.similarity_search(query, k=3)
    # Process docs to remove similar cases display
    return "\n".join([d.page_content.split("Similar Cases:")[0] for d in docs])

"""
Purpose:

Finds relevant historical CAPAs based on meaning (not just keywords)

Why Used:

Semantic over similarity: Understands context (e.g., "contamination" matches "particulate matter")

Filtering: Removes similar cases display per your requirements

Precision: k=3 returns only the most relevant cases to reduce noise


"""
# --------------------------
# Specialized Tools
# --------------------------
"""
Purpose:

Systematically identifies underlying causes using proven frameworks

Why Used:

5 Whys: Drills down to root cause (e.g., "Why? → Why? → Why?")

Fishbone: Categorizes causes (People/Methods/Materials/etc.)

Structured output: Returns ranked causes with confidence levels
"""
def rca_framework(description: str) -> str:
    """Root Cause Analysis framework tool"""
    prompt = ChatPromptTemplate.from_template(
        """Analyze this CAPA description using:
        1. 5 Whys technique
        2. Fishbone diagram categories
        3. Fault tree analysis
        
        CAPA Description: {description}
        
        Provide 3 probable root causes with confidence levels"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"description": description})



"""
Purpose:

Quantifies severity of the CAPA issue

Why Used:

Weighted factors: Prioritizes safety/regulatory impact

Normalized score (0-1): Allows comparison across CAPAs

Automated consistency: Removes subjective human bias
"""

def impact_scorer(description: str) -> float:
    """Impact scoring tool"""
    prompt = ChatPromptTemplate.from_template(
        """Rate the impact (0-1) of this CAPA:
        - Safety (40%)
        - Regulatory (30%)
        - Financial (20%)
        - Operational (10%)
        
        Description: {description}
        Respond ONLY with the score (0.00-1.00)"""
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return float(chain.invoke({"description": description}))
    except:
        return 0.5


"""
Purpose:

Captures user corrections to improve future analyses

Why Used:

Continuous learning: Updates vector embeddings with corrected root causes

Adaptive system: Becomes more accurate over time

Closed-loop: Implements your requirement for feedback integration

"""

def feedback_logger(feedback: str) -> str:
    """Tool for processing feedback"""
    # In production, would update ChromaDB embeddings
    return "Feedback recorded for system improvement"

# Create tools
tools = [
    Tool(
        name="SemanticSearchCAPA",
        func=semantic_search,
        description="Meaning-based search of CAPA knowledge base"
    ),
    Tool(
        name="RCAAnalysis",
        func=rca_framework,
        description="Structured root cause analysis"
    ),
    Tool(
        name="ImpactScorer",
        func=impact_scorer,
        description="Calculate CAPA impact score (0-1)"
    ),
    Tool(
        name="FeedbackHandler",
        func=feedback_logger,
        description="Process user feedback"
    )
]

# --------------------------
# LangGraph Workflow (Modified)
# --------------------------

class AgentState(TypedDict):
    description: str
    root_causes: List[str]
    impact_score: float
    feedback: Optional[str]
    needs_feedback: bool

def retrieve_context(state: AgentState):
    """Node: Semantic search"""
    context = semantic_search(state["description"])
    return {"context": context}

def analyze_causes(state: AgentState):
    """Node: Root cause analysis"""
    causes = rca_framework(state["description"])
    return {"root_causes": [causes]}  # Simplified parsing

def assess_impact(state: AgentState):
    """Node: Impact scoring"""
    score = impact_scorer(state["description"])
    return {"impact_score": score}

def handle_feedback(state: AgentState):
    """Node: Feedback processing"""
    if state.get("needs_feedback", False):
        feedback_logger(state["feedback"])
        return {"message": "System updated with feedback"}
    return {"message": "Analysis complete"}

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("analyze", analyze_causes)
workflow.add_node("impact", assess_impact)
workflow.add_node("feedback", handle_feedback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", "impact")
workflow.add_conditional_edges(
    "impact",
    lambda state: "feedback" if state.get("needs_feedback", False) else END,
    {"feedback": "feedback", END: END}
)
workflow.add_edge("feedback", END)

capa_workflow = workflow.compile()

# --------------------------
# User Interface
# --------------------------

def analyze_capa(description: str):
    """Public interface for CAPA analysis"""
    result = capa_workflow.invoke({
        "description": description,
        "needs_feedback": False
    })
    
    return {
        "root_causes": result["root_causes"],
        "impact_score": result["impact_score"]
    }

def submit_feedback(feedback: str):
    """Public interface for feedback"""
    feedback_logger(feedback)
    return {"status": "Feedback processed"}

# Example Usage
if __name__ == "__main__":
    analysis = analyze_capa("Product contamination in Batch #123")
    print("Root Causes:", analysis["root_causes"])
    print("Impact Score:", analysis["impact_score"])



++++++++++++++++++++++++++++++++++++++++
import os
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain import hub
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import numpy as np

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHROMA_DB_PATH = "RCA_db"  # Directory to store ChromaDB data

# --------------------------
# Step 1: PDF Loading and Processing
# --------------------------

def load_and_process_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Load and split PDFs into properly formatted documents"""
    all_documents = []
    
    for pdf_path in pdf_paths:
        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # 2. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        documents = text_splitter.split_documents(docs)
        all_documents.extend(documents)
    
    return all_documents

# --------------------------
# Step 2: Vector Store Initialization
# --------------------------

# Initialize Azure OpenAI components
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model_name="gpt-4",
    temperature=0.3
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model="text-embedding-ada-002"
)

# Initialize Chroma DB with persistence
def initialize_vector_store(pdf_paths: List[str]):
    """Initialize ChromaDB with PDF documents"""
    documents = load_and_process_pdfs(pdf_paths)
    
    chroma_settings = {
        "persist_directory": CHROMA_DB_PATH,
        "anonymized_telemetry": False
    }
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_settings["persist_directory"]
    )

# --------------------------
# Step 3: Semantic Search Tool
# --------------------------

def semantic_search(query: str, vector_db: Chroma) -> str:
    """Perform semantic search with meaning-based queries"""
    docs = vector_db.similarity_search(query, k=3)
    # Process docs to remove similar cases display
    return "\n".join([d.page_content.split("Similar Cases:")[0] for d in docs])

# --------------------------
# Step 4: Specialized Tools
# --------------------------

def rca_framework(description: str) -> str:
    """Root Cause Analysis framework tool"""
    prompt = ChatPromptTemplate.from_template(
        """Analyze this CAPA description using:
        1. 5 Whys technique
        2. Fishbone diagram categories
        3. Fault tree analysis
        
        CAPA Description: {description}
        
        Provide 3 probable root causes with confidence levels"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"description": description})

def impact_scorer(description: str) -> float:
    """Impact scoring tool"""
    prompt = ChatPromptTemplate.from_template(
        """Rate the impact (0-1) of this CAPA:
        - Safety (40%)
        - Regulatory (30%)
        - Financial (20%)
        - Operational (10%)
        
        Description: {description}
        Respond ONLY with the score (0.00-1.00)"""
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return float(chain.invoke({"description": description}))
    except:
        return 0.5  # Default if parsing fails

def feedback_logger(feedback: str) -> str:
    """Tool for processing feedback"""
    # In production, would update ChromaDB embeddings
    return "Feedback recorded for system improvement"

# --------------------------
# Step 5: Initialize System with PDF Data
# --------------------------

# Example usage - initialize with your PDF files
pdf_paths = ["path/to/capa1.pdf", "path/to/capa2.pdf"]  # Replace with actual paths
vector_db = initialize_vector_store(pdf_paths)

# Create tools
tools = [
    Tool(
        name="SemanticSearchCAPA",
        func=lambda q: semantic_search(q, vector_db),
        description="Meaning-based search of CAPA knowledge base"
    ),
    Tool(
        name="RCAAnalysis",
        func=rca_framework,
        description="Structured root cause analysis"
    ),
    Tool(
        name="ImpactScorer",
        func=impact_scorer,
        description="Calculate CAPA impact score (0-1)"
    ),
    Tool(
        name="FeedbackHandler",
        func=feedback_logger,
        description="Process user feedback"
    )
]

# --------------------------
# Step 6: LangGraph Workflow
# --------------------------

class AgentState(TypedDict):
    description: str
    root_causes: List[str]
    impact_score: float
    feedback: Optional[str]
    needs_feedback: bool

def retrieve_context(state: AgentState):
    """Node: Semantic search"""
    context = semantic_search(state["description"], vector_db)
    return {"context": context}

def analyze_causes(state: AgentState):
    """Node: Root cause analysis"""
    causes = rca_framework(state["description"])
    return {"root_causes": [causes]}  # Simplified parsing

def assess_impact(state: AgentState):
    """Node: Impact scoring"""
    score = impact_scorer(state["description"])
    return {"impact_score": score}

def handle_feedback(state: AgentState):
    """Node: Feedback processing"""
    if state.get("needs_feedback", False):
        feedback_logger(state["feedback"])
        return {"message": "System updated with feedback"}
    return {"message": "Analysis complete"}

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("analyze", analyze_causes)
workflow.add_node("impact", assess_impact)
workflow.add_node("feedback", handle_feedback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", "impact")
workflow.add_conditional_edges(
    "impact",
    lambda state: "feedback" if state.get("needs_feedback", False) else END,
    {"feedback": "feedback", END: END}
)
workflow.add_edge("feedback", END)

capa_workflow = workflow.compile()

# --------------------------
# Step 7: User Interface
# --------------------------

def analyze_capa(description: str):
    """Public interface for CAPA analysis"""
    result = capa_workflow.invoke({
        "description": description,
        "needs_feedback": False
    })
    
    return {
        "root_causes": result["root_causes"],
        "impact_score": result["impact_score"]
    }

def submit_feedback(feedback: str):
    """Public interface for feedback"""
    feedback_logger(feedback)
    return {"status": "Feedback processed"}

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    # Initialize with PDFs (run once)
    # pdf_paths = ["capa_reports/report1.pdf", ...]
    # vector_db = initialize_vector_store(pdf_paths)
    
    # Analyze a new case
    analysis = analyze_capa("Product contamination in Batch #123")
    print("Root Causes:", analysis["root_causes"])
    print("Impact Score:", analysis["impact_score"])
    
    # Submit feedback
    feedback_result = submit_feedback("Root cause was actually humidity controls")
    print(feedback_result)
