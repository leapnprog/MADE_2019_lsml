package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.StructType

trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasPredictionCol with HasStepSize with HasMaxIter {

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkNumericType(schema, getLabelCol)

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      // schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
    schema
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  def setStepSize(value: Double): this.type = set(stepSize, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  setDefault(stepSize -> 0.1, maxIter -> 1000)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val extFeatures = dataset.withColumn("auxiliary_feature", lit(1))
    val assembler = new VectorAssembler()
      .setInputCols(Array("auxiliary_feature", $(featuresCol), $(labelCol)))
      .setOutputCol("assembled_features")
    val features: Dataset[Vector] = assembler
      .transform(extFeatures)
      .select("assembled_features").as[Vector]

    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights = breeze.linalg.DenseVector.rand[Double](numFeatures + 1)

    for (i <- 0 until $(maxIter)) {
      val summary = features.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(sample => {
          val x = sample.asBreeze(0 until (numFeatures + 1)).toDenseVector
          val y = sample.asBreeze(-1)
          val grad = x * (breeze.linalg.sum(x * weights) - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(grad))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      weights -= $(stepSize) * summary.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(weights(1 until (numFeatures + 1))).toDense,
      weights(0)
    )).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val weights: DenseVector,
                                           val bias: Double) extends Model[LinearRegressionModel]
  with LinearRegressionParams with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights, bias))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val breezeWeights = weights.asBreeze
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x: Vector) => {
        Vectors.dense((x.asBreeze dot breezeWeights) + bias)
      })

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val parameters = weights.asInstanceOf[Vector] -> Vectors.dense(bias)

      sqlContext.createDataFrame(Seq(parameters)).write.parquet(path + "/parameters")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {

  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val parameters = sqlContext.read.parquet(path + "/parameters")

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val (weights, bias) = parameters.select(parameters("_1").as[Vector], parameters("_2").as[Vector]).first()

      val model = new LinearRegressionModel(weights.toDense, bias(0))
      metadata.getAndSetParams(model)
      model
    }
  }
}
