package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.01
  val stepSize = 0.1
  val maxIter = 1000

  lazy val weights: DenseVector[Double] = LinearRegressionTest._weights
  lazy val bias: Double = LinearRegressionTest._bias

  lazy val X: DenseMatrix[Double] = LinearRegressionTest._X
  lazy val y: DenseVector[Double] = LinearRegressionTest._y
  lazy val data: DataFrame = LinearRegressionTest._data


  "Model" should "calculate regression" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(weights).toDense,
      bias = bias
    ).setFeaturesCol("features")
      .setLabelCol("target")
      .setPredictionCol("prediction")

    val regression: Array[Double] = model.transform(data).collect().map(_.getAs[Vector](2)(0))

    regression.length should be(y.length)

    for (i <- regression.indices) {
      regression(i) should be(y(i) +- delta)
    }
  }

  "Estimator" should "calculate parameters" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setPredictionCol("prediction")
      .setStepSize(stepSize)
      .setMaxIter(maxIter)

    val model = estimator.fit(data)

    model.weights(0) should be(weights(0) +- delta)
    model.weights(1) should be(weights(1) +- delta)
    model.weights(2) should be(weights(2) +- delta)
    model.bias should be(bias +- delta)
  }

  "Estimator" should "produce functional model" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setPredictionCol("prediction")
      .setStepSize(stepSize)
      .setMaxIter(maxIter)

    val model = estimator.fit(data)

    val regression: Array[Double] = model.transform(data).collect().map(_.getAs[Vector](2)(0))

    regression.length should be(y.length)

    for (i <- regression.indices) {
      regression(i) should be(y(i) +- delta)
    }
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("target")
        .setPredictionCol("prediction")
        .setStepSize(stepSize)
        .setMaxIter(maxIter)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    model.weights(0) should be(weights(0) +- delta)
    model.weights(1) should be(weights(1) +- delta)
    model.weights(2) should be(weights(2) +- delta)
    model.bias should be(bias +- delta)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("target")
        .setPredictionCol("prediction")
        .setStepSize(stepSize)
        .setMaxIter(maxIter)
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    val regression: Array[Double] = reRead.transform(data).collect().map(_.getAs[Vector](2)(0))

    regression.length should be(y.length)

    for (i <- regression.indices) {
      regression(i) should be(y(i) +- delta)
    }
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _weights: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
  lazy val _bias: Double = 1.0

  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](100000, 3)
  lazy val _y: DenseVector[Double] = _X * _weights + _bias

  lazy val _dataBreeze: DenseMatrix[Double] = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)
  lazy val _data: DataFrame = {
    import sqlc.implicits._

    val tmpData = _dataBreeze(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "target")

    val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")

    assembler.transform(tmpData).select("features", "target")
  }
}
