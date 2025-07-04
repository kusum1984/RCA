SELECT
  *,
  COUNT(`Delivery Lines GT0`) OVER () AS `Count of Delivery Lines GT0`
FROM (
  SELECT
    -- ADRC
    adrc.COUNTRY AS `Vendor Country`,

    -- EKET
    eket.CHARG AS `Batch`,
    eket.EINDT AS `Delivery Date`,
    eket.GLMNG AS `Delivery Quantity Unit`,

    -- EKKO
    ekko.AEDAT AS `Delivery Created Date`,

    -- EKPO
    ekpo.MATNR AS `Material`,
    ekpo.WERKS AS `Plant`,

    -- LFA1
    lfa1.LIFNR AS `Vendor`,
    lfa1.NAME1 AS `Vendor Name`,

    -- T001W
    t001w.NAME1 AS `Plant Name`,

    -- LIPS via VBFA
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

    CURRENT_DATE() AS `Report Date`,
    CURRENT_TIMESTAMP() AS `Report Time`,

    DATEDIFF(CURRENT_DATE(), eket.EINDT) AS `DaysPastDue`

  FROM
    EKPO ekpo

  -- Join to EKET (Schedule lines)
  LEFT JOIN EKET eket
    ON ekpo.EBELN = eket.EBELN AND ekpo.EBELP = eket.EBELP

  -- Join to EKKO (PO header)
  LEFT JOIN EKKO ekko
    ON ekpo.EBELN = ekko.EBELN

  -- Join to LFA1 (Vendor master)
  LEFT JOIN LFA1 lfa1
    ON ekko.LIFNR = lfa1.LIFNR

  -- Join to ADRC (Address, for country)
  LEFT JOIN ADRC adrc
    ON lfa1.ADRNR = adrc.ADRNR

  -- Join to T001W (Plant text)
  LEFT JOIN T001W t001w
    ON ekpo.WERKS = t001w.WERKS

  -- Use VBFA to link EKPO to delivery (LIPS)
  LEFT JOIN VBFA vbfa
    ON ekpo.EBELN = vbfa.VBELV
    AND ekpo.EBELP = vbfa.POSNV
    AND vbfa.VBTYP_N = 'J'  -- delivery document

  -- Join to LIPS (delivery items)
  LEFT JOIN LIPS lips
    ON vbfa.VBELN = lips.VBELN
    AND vbfa.POSNN = lips.POSNR
    AND lips.WBSTA = 'C'  -- GI completed only

  -- Join to LIKP (delivery header)
  LEFT JOIN LIKP likp
    ON lips.VBELN = likp.VBELN

  -- Final filter
  WHERE ekpo.WERKS = 'CA01'
) AS subquery
