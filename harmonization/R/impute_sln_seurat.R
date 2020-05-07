# Code adapted from https://github.com/satijalab/Integration2019/blob/master/analysis_code/citeseq/cite_cross_validations.R

library(Seurat)
set.seed(1234)

# setwd("/data/yosef2/users/adamgayoso/projects/totalVI_journal")
setwd("~/Google Drive/Berkeley/Yosef_lab/totalVI_journal")
umis_sln_d1 = read.csv('data/raw_data/sln_d1_harmo_rna_seurat.csv.gz', row.names = 1)
umis_sln_d2 = read.csv('data/raw_data/sln_d2_harmo_rna_seurat.csv.gz', row.names = 1)
adts = read.csv('data/raw_data/sln_d1_harmo_pro_seurat.csv.gz', row.names = 1)

sln_d1 <- CreateSeuratObject(counts = umis_sln_d1)
sln_d1[["ADT"]] <- CreateAssayObject(counts = adts)

DefaultAssay(object = sln_d1) <- "RNA"
sln_d1 <- NormalizeData(object = sln_d1)
sln_d1 <- ScaleData(object = sln_d1)
features <- rownames(sln_d1)

sln_d2 <- CreateSeuratObject(counts = umis_sln_d2)
DefaultAssay(object = sln_d2) <- "RNA"
sln_d2 <- NormalizeData(object = sln_d2)
sln_d2 <- ScaleData(object = sln_d2)

transfer.anchors <- FindTransferAnchors(
  reference = sln_d1,
  query = sln_d2,
  features = features,
  dims = 1:50,
  npcs = 50,
  verbose = TRUE
)

refdata <- GetAssayData(
  object = sln_d1,
  assay = 'ADT',
  slot = 'data'
)

imputed.data <- TransferData(
  anchorset = transfer.anchors,
  refdata = refdata,
  l2.norm = FALSE,
  dims = 1:50,
  k.weight = 50,
  slot = 'counts'
)

write.csv(imputed.data@data, file="harmonization/seurat_harmo_results/imputed_sln.csv")
