from ..node import Stream

# Session Information (rephro-ephys)
eids = [
    'db4df448-e449-4a6f-a0e7-288711e7a75a',  # Berkeley
    'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',  # Berkeley
    '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',  # Berkeley
    'e535fb62-e245-4a48-b119-88ce62a6fe67',  # Berkeley
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # Berkeley
    'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',  # Berkeley
    '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',  # CCU
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe',  # CCU
    'a4a74102-2af5-45dc-9e41-ef7f5aed88be',  # CCU
    '746d1902-fa59-4cab-b0aa-013be36060d5',  # CCU
    '88224abb-5746-431f-9c17-17d7ef806e6a',  # CCU
    '0802ced5-33a3-405e-8336-b65ebc5cb07c',  # CCU
    'ee40aece-cffd-4edb-a4b6-155f158c666a',  # CCU
    'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',  # CCU
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # CCU
    'dda5fc59-f09a-4256-9fb5-66c67667a466',  # CSHL(C)
    '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',  # CSHL(C)
    'f312aaec-3b6f-44b3-86b4-3a0c119c0438',  # CSHL(C)
    '4b00df29-3769-43be-bb40-128b1cba6d35',  # CSHL(C)
    'ecb5520d-1358-434c-95ec-93687ecd1396',  # CSHL(C)
    '51e53aff-1d5d-4182-a684-aba783d50ae5',  # NYU
    'f140a2ec-fd49-4814-994a-fe3476f14e66',  # NYU
    'a8a8af78-16de-4841-ab07-fde4b5281a03',  # NYU
    '61e11a11-ab65-48fb-ae08-3cb80662e5d6',  # NYU
    '73918ae1-e4fd-4c18-b132-00cb555b1ad2',  # Princeton
    'd9f0c293-df4c-410a-846d-842e47c6b502',  # Princeton
    'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # SWC(H)
    '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',  # SWC(H)
    '56b57c38-2699-4091-90a8-aba35103155e',  # SWC(M)
    '3638d102-e8b6-4230-8742-e548cd87a949',  # SWC(M)
    '7cb81727-2097-4b52-b480-c89867b5b34c',  # SWC(M)
    '781b35fd-e1f0-4d14-b2bb-95b7263082bb',  # UCL
    '3f859b5c-e73a-4044-b49e-34bb81e96715',  # UCL
    'b22f694e-4a34-4142-ab9d-2556c3487086',  # UCL
    '0a018f12-ee06-4b11-97aa-bbbff5448e9f',  # UCL
    'aad23144-0e52-4eac-80c5-c4ee2decb198',  # UCL
    'b196a2ad-511b-4e90-ac99-b5a29ad25c22',  # UCL
    'e45481fa-be22-4365-972c-e7404ed8ab5a',  # UCL
    'd04feec7-d0b7-4f35-af89-0232dd975bf0',  # UCL
    '1b715600-0cbc-442c-bd00-5b0ac2865de1',  # UCL
    'c7bf2d49-4937-4597-b307-9f39cb1c7b16',  # UCL
    '8928f98a-b411-497e-aa4b-aa752434686d',  # UCL
    'ebce500b-c530-47de-8cb1-963c552703ea',  # UCLA
    'dc962048-89bb-4e6a-96a9-b062a2be1426',  # UCLA
    '6899a67d-2e53-4215-a52a-c7021b5da5d4',  # UCLA
    '15b69921-d471-4ded-8814-2adad954bcd8',  # UCLA
    '5ae68c54-2897-4d3a-8120-426150704385',  # UCLA
    'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53',  # UCLA
    '824cf03d-4012-4ab1-b499-c83a92c5589e',  # UCLA
    '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca',  # UW
    'f115196e-8dfe-4d2a-8af3-8206d93c1729',  # UW
    '9b528ad0-4599-4a55-9148-96cc1d93fb24',  # UW
    '3e6a97d3-3991-49e2-b346-6948cb4580fb',  # UW
]

dropbox_marker_paths = {
    'db4df448-e449-4a6f-a0e7-288711e7a75a': 'danlab_DY_009_2020-02-27-001',
    'd23a44ef-1402-4ed7-97f5-47e9a7a504d9': 'danlab_DY_016_2020-09-12-001',
    '4a45c8ba-db6f-4f11-9403-56e06a33dfa4': 'danlab_DY_020_2020-09-29-001',
    'e535fb62-e245-4a48-b119-88ce62a6fe67': 'danlab_DY_013_2020-03-12-001',
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce': 'danlab_DY_018_2020-10-15-001',
    'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0': 'danlab_DY_010_2020-01-27-001',
    '30c4e2ab-dffc-499d-aae4-e51d6b3218c2': 'mainenlab_ZFM-02370_2021-04-29-001',
    'd0ea3148-948d-4817-94f8-dcaf2342bbbe': 'mainenlab_ZFM-01936_2021-01-19-001',
    'a4a74102-2af5-45dc-9e41-ef7f5aed88be': 'mainenlab_ZFM-02368_2021-06-01-001',
    '746d1902-fa59-4cab-b0aa-013be36060d5': 'mainenlab_ZFM-01592_2020-10-20-001',
    '88224abb-5746-431f-9c17-17d7ef806e6a': 'mainenlab_ZFM-02372_2021-06-01-002',
    '0802ced5-33a3-405e-8336-b65ebc5cb07c': 'mainenlab_ZFM-02373_2021-06-23-001',
    'ee40aece-cffd-4edb-a4b6-155f158c666a': 'mainenlab_ZM_2241_2020-01-30-001',
    'c7248e09-8c0d-40f2-9eb4-700a8973d8c8': 'mainenlab_ZM_3001_2020-08-05-001',
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02': 'mainenlab_ZFM-02369_2021-05-19-001',
    'dda5fc59-f09a-4256-9fb5-66c67667a466': 'churchlandlab_CSHL059_2020-03-06-001',
    '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b': 'churchlandlab_CSHL049_2020-01-08-001',
    'f312aaec-3b6f-44b3-86b4-3a0c119c0438': 'churchlandlab_CSHL058_2020-07-07-001',
    '4b00df29-3769-43be-bb40-128b1cba6d35': 'churchlandlab_CSHL052_2020-02-21-001',
    'ecb5520d-1358-434c-95ec-93687ecd1396': 'churchlandlab_CSHL051_2020-02-05-001',
    '51e53aff-1d5d-4182-a684-aba783d50ae5': 'angelakilab_NYU-45_2021-07-19-001',
    'f140a2ec-fd49-4814-994a-fe3476f14e66': 'angelakilab_NYU-47_2021-06-21-003',
    'a8a8af78-16de-4841-ab07-fde4b5281a03': 'angelakilab_NYU-12_2020-01-22-001',
    '61e11a11-ab65-48fb-ae08-3cb80662e5d6': 'angelakilab_NYU-21_2020-08-10-002',
    '73918ae1-e4fd-4c18-b132-00cb555b1ad2': 'wittenlab_ibl_witten_27_2021-01-21-001',
    'd9f0c293-df4c-410a-846d-842e47c6b502': 'wittenlab_ibl_witten_25_2020-12-07-002',
    'dac3a4c1-b666-4de0-87e8-8c514483cacf': 'hoferlab_SWC_060_2020-11-24-001',
    '6f09ba7e-e3ce-44b0-932b-c003fb44fb89': 'hoferlab_SWC_043_2020-09-16-002',
    '56b57c38-2699-4091-90a8-aba35103155e': 'mrsicflogellab_SWC_054_2020-10-05-001',
    '3638d102-e8b6-4230-8742-e548cd87a949': 'mrsicflogellab_SWC_058_2020-12-07-001',
    '7cb81727-2097-4b52-b480-c89867b5b34c': 'mrsicflogellab_SWC_052_2020-10-22-001',
    '781b35fd-e1f0-4d14-b2bb-95b7263082bb': 'cortexlab_KS044_2020-12-09-001',
    '3f859b5c-e73a-4044-b49e-34bb81e96715': 'cortexlab_KS094_2022-06-17-001',
    'b22f694e-4a34-4142-ab9d-2556c3487086': 'cortexlab_KS055_2021-05-02-001',
    '0a018f12-ee06-4b11-97aa-bbbff5448e9f': 'cortexlab_KS051_2021-05-11-001',
    'aad23144-0e52-4eac-80c5-c4ee2decb198': 'cortexlab_KS023_2019-12-10-001',
    'b196a2ad-511b-4e90-ac99-b5a29ad25c22': 'cortexlab_KS084_2022-02-01-001',
    'e45481fa-be22-4365-972c-e7404ed8ab5a': 'cortexlab_KS086_2022-03-15-001',
    'd04feec7-d0b7-4f35-af89-0232dd975bf0': 'cortexlab_KS089_2022-03-19-001',
    '1b715600-0cbc-442c-bd00-5b0ac2865de1': 'cortexlab_KS084_2022-01-31-001',
    'c7bf2d49-4937-4597-b307-9f39cb1c7b16': 'cortexlab_KS074_2021-11-22-001',
    '8928f98a-b411-497e-aa4b-aa752434686d': 'cortexlab_KS096_2022-06-17-001',
    'ebce500b-c530-47de-8cb1-963c552703ea': 'churchlandlab_ucla_MFD_09_2023-10-19-001',
    'dc962048-89bb-4e6a-96a9-b062a2be1426': 'churchlandlab_ucla_UCLA048_2022-08-16-001',
    '6899a67d-2e53-4215-a52a-c7021b5da5d4': 'churchlandlab_ucla_MFD_06_2023-08-29-001',
    '15b69921-d471-4ded-8814-2adad954bcd8': 'churchlandlab_ucla_MFD_07_2023-08-31-001',
    '5ae68c54-2897-4d3a-8120-426150704385': 'churchlandlab_ucla_MFD_08_2023-09-07-001',
    'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53': 'churchlandlab_ucla_MFD_05_2023-08-16-001',
    '824cf03d-4012-4ab1-b499-c83a92c5589e': 'churchlandlab_ucla_UCLA011_2021-07-20-001',
    '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca': 'steinmetzlab_NR_0024_2023-01-19-001',
    'f115196e-8dfe-4d2a-8af3-8206d93c1729': 'steinmetzlab_NR_0021_2022-06-23-003',
    '9b528ad0-4599-4a55-9148-96cc1d93fb24': 'steinmetzlab_NR_0019_2022-04-29-001',
    '3e6a97d3-3991-49e2-b346-6948cb4580fb': 'steinmetzlab_NR_0020_2022-05-08-001',
    #dropbox follows
    '46794e05-3f6a-4d35-afb3-9165091a5a74': 'churchlandlab_CSHL045_2020-02-27-001',
    'db4df448-e449-4a6f-a0e7-288711e7a75a': 'danlab_DY_009_2020-02-27-001',
    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce': 'danlab_DY_018_2020-10-15-001',
    'f3ce3197-d534-4618-bf81-b687555d1883': 'hoferlab_SWC_043_2020-09-15-001',
    '493170a6-fd94-4ee4-884f-cc018c17eeb9': 'hoferlab_SWC_061_2020-11-23-001',
    '7cb81727-2097-4b52-b480-c89867b5b34c': 'mrsicflogellab_SWC_052_2020-10-22-001',
    'ff96bfe1-d925-4553-94b5-bf8297adf259': 'wittenlab_ibl_witten_26_2021-01-27-002',
    '73918ae1-e4fd-4c18-b132-00cb555b1ad2': 'wittenlab_ibl_witten_27_2021-01-21-001'
}

class IBLEidStream(Stream):
    def __init__(self, label: str, idx) -> None:
        super().__init__(label)
        
        if type(idx) == int:
            self.data[label] = { 'eid': eids[idx], 'sess': dropbox_marker_paths[eids[idx]] }
        else:
            self.data[label] = { 'eid': idx, 'sess': dropbox_marker_paths[idx] }