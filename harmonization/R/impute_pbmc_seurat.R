# Code adapted from https://github.com/satijalab/Integration2019/blob/master/analysis_code/citeseq/cite_cross_validations.R

library(Seurat)
set.seed(1234)

# setwd("/data/yosef2/users/adamgayoso/projects/totalVI_journal")
setwd("~/Google Drive/Berkeley/Yosef_lab/totalVI_journal")
umis_10k = read.csv('data/raw_data/pbmc10k_harmo_rna.csv.gz', row.names = 1)
umis_5k = read.csv('data/raw_data/pbmc5k_harmo_rna.csv.gz', row.names = 1)
adts = read.csv('data/raw_data/pbmc10k_harmo_pro.csv.gz', row.names = 1)

pbmc_10k <- CreateSeuratObject(counts = umis_10k)
pbmc_10k[["ADT"]] <- CreateAssayObject(counts = adts)

DefaultAssay(object = pbmc_10k) <- "RNA"
pbmc_10k <- NormalizeData(object = pbmc_10k)
pbmc_10k <- ScaleData(object = pbmc_10k)
features <- rownames(pbmc_10k)

pbmc_5k <- CreateSeuratObject(counts = umis_5k)
DefaultAssay(object = pbmc_5k) <- "RNA"
pbmc_5k <- NormalizeData(object = pbmc_5k)
pbmc_5k <- ScaleData(object = pbmc_5k)

transfor.anchors <- FindTransferAnchors(
  reference = pbmc_10k,
  query = pbmc_5k,
  features = features,
  dims = 1:50,
  npcs = 50,
  verbose = TRUE
)

refdata <- GetAssayData(
  object = pbmc_10k,
  assay = 'ADT',
  slot = 'data'
)

imputed.data <- TransferData(
  anchorset = transfor.anchors,
  refdata = refdata,
  l2.norm = FALSE,
  dims = 1:50,
  k.weight = 50,
  slot = 'counts'
)

write.csv(imputed.data@data, file="harmonization/seurat_harmo_results/imputed_pbmc.csv")

