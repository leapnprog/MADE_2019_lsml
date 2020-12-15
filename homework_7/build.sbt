name := "LinReg_on_Spark"

version := "0.1"

scalaVersion := "2.12.10"

val sparkVersion = "3.0.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.scalatest" %% "scalatest" % "3.2.2" % "test" withSources()
)
