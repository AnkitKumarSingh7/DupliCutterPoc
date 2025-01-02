# codingStandards.py

config = {
    'general': {
        'maxFunctionLines': 20,
        'maxFileLines': 500,
        'avoidAnyType': True,
        'avoidVar': True,
        'preferConstOverLet': True,
    },
    'namingConventions': {
        'typeNames': 'PascalCase',
        'functionNames': 'camelCase',
        'localVariables': 'camelCase',
        'privateProperties': '_camelCase',
        'importOrder': ['thirdParty', 'companyModules', 'localModules'],
        'avoidUnusedImports': True,
        'avoidRequireImport': True,
    },
    'errorHandling': {
        'wrapAsyncInTryCatch': True,
        'meaningfulErrorMessages': True,
        'logUnexpectedErrors': True,
        'logKnownFailures': True,
    },
    'reactGuidelines': {
        'preferFunctionalComponents': True,
        'singleResponsibility': True,
        'defineExplicitInterfaces': True,
        'optimizeRendering': True,
    },
    'azurePortal': {
        'networkRequestNaming': 'DataFetcher',
        'useDataCache': True,
    },
    'redundancyCheck': {
        'threshold': 0.8,  # For similarity of code to trigger a suggestion to refactor
    },
}
