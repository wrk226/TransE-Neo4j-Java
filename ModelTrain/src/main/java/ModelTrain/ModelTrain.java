package ModelTrain;

import org.apache.commons.lang3.ArrayUtils;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.factory.GraphDatabaseFactory;
import org.neo4j.unsafe.batchinsert.BatchInserter;
import org.neo4j.unsafe.batchinsert.BatchInserters;

import java.io.File;
import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public class ModelTrain {
    static Map<String, Long> relation2id = new HashMap<>();
    static Map<String, Long> relation2node = new HashMap<>();
    public static void main(String[] args) throws Exception {
        // C:\Users\D-blue\Desktop\Graph_Database\hw6\db\FB13 100 5000 1 1 l1 9999 25 C:\Users\D-blue\Desktop\Graph_Database\hw6\KGC_datasets\FB13\relation2id.txt FB13_alpha_0.001_gamma_1.0_bs_5000_me_80_nr_25_MR.model
        final String neo4jFolder = args[0];
        String[] split=neo4jFolder.split("\\\\");
        String datasetName=split[split.length-1];
        System.out.println("data="+datasetName);
        final int dimension = Integer.parseInt(args[1]); //100
        System.out.println("dimension="+args[1]);
        final int batchSize = Integer.parseInt(args[2]); //5000
        System.out.println("batchSize="+args[2]);
        final double alpha = Double.parseDouble(args[3]);
        System.out.println("alpha="+args[3]);
        final double gamma = Double.parseDouble(args[4]);
        System.out.println("gamma="+args[4]);
        String norm = args[5];
        System.out.println("norm="+args[5]);
        final int maxEpochs=Integer.parseInt(args[6]);
        final int negativeRate=Integer.parseInt(args[7]); //25
        System.out.println("negativeRate="+args[7]);
        final String relation2id_file=args[8];
        String input_file=null;
        boolean load_file=false;
        if(args.length>9){
            input_file=args[9];
            load_file=true;
            System.out.println("load file detected, will load model "+input_file);
        }
        // number of times allow the score increase
        final int patient=3;
        // number of epochs between each evaluate
        final int skip_epoch=100;
        // number of epoch before first evaluate
        int start_epoch=0;
        // subset of validation data in evaluation,-1 if want to use all validation data
        int validation_subset=-1; //2500
        // number of epoch in loaded model, don't change it
        int epoch_init=0;

        // Read predicates ids.
        Scanner sc = new Scanner(new File(relation2id_file));
        while (sc.hasNextLine()) {
            String[] line = sc.nextLine().split("\t");
            if (line.length == 2) {
                relation2id.put(line[0], Long.valueOf(line[1]));
            }
        }
        sc.close();
        // set up database
        GraphDatabaseService db = new GraphDatabaseFactory().newEmbeddedDatabase(new File(neo4jFolder));
        Transaction tx=db.beginTx();
        // save current split
        System.out.println("Saving current split...");
        saveSplits(db,datasetName);
        // select subset of validation
        if (validation_subset!=-1){
            db.execute("match ()-[p]->() where p.split='Validation' with rand() as r, p order by r skip "+validation_subset+" delete p");
        }
        // load model
        if(load_file){
            String file=new File(input_file).getName();
            epoch_init=Integer.parseInt(file.split("_")[8]);
            System.out.println("Loading model embedding...");
            norm="l"+loadFromFile(db,input_file);
        }
        // assign object and subject for each of entities
        LCWA(db);
        Result rs;
        // init parameters
        Random rand = new Random(123);
        double init_min = -6 / Math.sqrt(dimension);
        double init_max = 6 / Math.sqrt(dimension);
        if(!load_file){
            System.out.println("Initializing entity embeddings...");
            // initialize entity embedding
            rs = db.execute("match (e)-[{split:'Training'}]-() return distinct(id(e)) as eid");
            while (rs.hasNext()) {
                db.getNodeById(Long.parseLong(rs.next().get("eid").toString()))
                        .setProperty("embedding", initialize(new double[dimension], rand, init_min, init_max));
            }
        }
        // initialize and normalize predicate embedding
        rs = db.execute("match ()-[p]->() return type(p) as ptype, min(id(p)) as pid");//
        // (27)-[_has_part,82]->(17736)
        String[] triple;
        while(rs.hasNext()){
            Map<String,Object> content=rs.next();
            relation2node.put(content.get("ptype").toString(),Long.parseLong(content.get("pid").toString()));
            if(!load_file){
                System.out.println("Initializing predicate embeddings...");
                db.getRelationshipById(Long.parseLong(content.get("pid").toString()))
                        .setProperty("embedding",normalize(initialize(new double[dimension],rand,init_min,init_max)));
            }

        }
        System.out.println("Training start!");
        // start epoch
        Double previous_score=null;
        Map<String,Object> content;

        int patient_count=0;
        for(int epoch=0;epoch<=maxEpochs;epoch++){
            long current_time=System.currentTimeMillis();
            // normalize entity embedding
            rs=db.execute("match (e)-[{split:'Training'}]-() return distinct(id(e)) as eid, e.embedding as e_emb");
            while(rs.hasNext()){
                content=rs.next();
                db.getNodeById(Long.parseLong(content.get("eid").toString()))
                        .setProperty("embedding",normalize((double[])content.get("e_emb")));
            }
            // get random batch
            rs=db.execute("match ()-[p{split:'Training'}]->() return distinct p as triple, rand() as r order by r limit $batchSize ",Map.of("batchSize",batchSize));
            while(rs.hasNext()){
                // [, 5786, -, _derivationally_related_form, 19952]->, 29110]
                triple=rs.next().get("triple").toString().split("[()\\[,]");
                long s=Long.parseLong(triple[1]);
                long p=relation2node.get(triple[3]);
                long o=Long.parseLong(triple[5]);
                // get corrupt subject and object
                Map<String,long[]> corrupts= corrupt(s, relation2id.get(triple[3]),o,db,negativeRate,rand,"train");
                // gradient descent on corrupt subject
                long[] corrupt_subject=corrupts.get("corrupt_subject");
                for(long sp:corrupt_subject){
                    long op=o;
                    gradient_descent(db,s,p,o,sp,op,norm,alpha,gamma);
                }
                // gradient descent on object
                long[] corrupt_object=corrupts.get("corrupt_object");
                for(long op:corrupt_object){
                    long sp=s;
                    gradient_descent(db,s,p,o,sp,op,norm,alpha,gamma);
                }
            }
            System.out.println("Epoch "+(epoch+epoch_init)+" | Time of epoch="+(System.currentTimeMillis()-current_time)/1000+"s");
            current_time=System.currentTimeMillis();
            // evaluate
            if(
                    epoch>=start_epoch &&
                    epoch%skip_epoch==0) {
                System.out.println("Start evaluating...");
                double score = evaluate(db, norm, "stop");
                System.out.println("Time of evaluation=" + (System.currentTimeMillis() - current_time) / 1000 + "s");
                System.out.println("MR score=" + score);

                //  patient=3
                if (previous_score != null && score > previous_score) {
                    if (patient_count < patient) {
                        patient_count++;
                        System.out.println("Patient="+patient_count);
                    } else {
                        System.out.println("Early stopped!");
                        break;
                    }
                } else {
                    previous_score = score;
                    patient_count=0;
                    saveToFile(db,Long.parseLong(norm.substring(1)),datasetName+"_alpha_"+alpha+"_gamma_"+gamma+"_bs_"+batchSize+"_me_"+(epoch+epoch_init)+"_nr_"+negativeRate+"_MR.model");
//                    saveToFile(db,Long.parseLong(norm.substring(1)),datasetName+"_alpha_"+alpha+"_gamma_"+gamma+"_bs_"+batchSize+"_me_"+maxEpochs+"_nr_"+negativeRate+"_MR.model");

                }

            }
        }
        // final test
        long current_time=System.currentTimeMillis();
        double score=evaluate(db,norm,"acc");
        System.out.println("Time of testing="+(System.currentTimeMillis()-current_time)/1000+"s");
        System.out.println("MR score="+score);
        // need to change the early stop criteria
        System.out.println("Finished!");
        rs.close();
        tx.close();
        db.shutdown();
    }
    static double[] initialize(double[] embedding,Random rand,double min,double max){
        // -6/sqrt(dim)<init<6/sqrt(dim)
        embedding= Arrays.stream(embedding).map(item->min+(max-min)*rand.nextDouble()).toArray();
        return embedding;
    }
    static double[] normalize(double[] embedding){
        double l2_distance = Math.sqrt(Arrays.stream(embedding).map(item -> Math.pow(item, 2)).sum());
        embedding=Arrays.stream(embedding).map(item->item/l2_distance).toArray();
        return embedding;
    }

    static void LCWA(GraphDatabaseService db){
        System.out.println("Generating object and subject for each entities...");
        // Use the predicate ids to create the properties, e.g., subjects_1_train, which are the subject in the training split
        //	for predicate 1.
        ArrayList<String[]> datasets=new ArrayList<>();
        datasets.add(new String[]{"Training"});
        datasets.add(new String[]{"Training","Validation"});
        datasets.add(new String[]{"Training","Validation","Test"});
        String[] dataset_name=new String[]{"train","valid","test"};
        for(int db_i=0;db_i<datasets.size();db_i++) {
            Map<String, Object> param = new HashMap<>();
            param.put("split", datasets.get(db_i));
            // get subjects_p_
            Result rs = db.execute("match (s)-[p]->(n) where p.split in $split return type(p) as p_type, collect(id(s)) as sids, id(n) as nid", param);
            while (rs.hasNext()) {
                Map<String, Object> content = rs.next();
                long p = relation2id.get(content.get("p_type").toString());
                ArrayList<?> sids = (ArrayList<?>) content.get("sids");
                long[] sid_list=new long[sids.size()];
                for(int index=0;index<sids.size();index++){
                    sid_list[index]=(long)sids.get(index);
                }
                int n = Integer.parseInt(content.get("nid").toString());
                db.getNodeById(n).setProperty("subjects_" + p + "_"+dataset_name[db_i], sid_list);
            }
            rs.close();
            // get objects_p_
            rs = db.execute("match (n)-[p]->(o) where p.split in $split return type(p) as p_type, collect(id(o)) as oids, id(n) as nid", param);
            while (rs.hasNext()) {
                Map<String, Object> content = rs.next();
                long p = relation2id.get(content.get("p_type").toString());
                ArrayList<?> oids = (ArrayList<?>) content.get("oids");
                long[] oid_list=new long[oids.size()];
                for(int index=0;index<oids.size();index++){
                    oid_list[index]=(long)oids.get(index);
                }
                int n = Integer.parseInt(content.get("nid").toString());
                db.getNodeById(n).setProperty("objects_" + p + "_"+dataset_name[db_i], oid_list);
            }
            rs.close();
        }
    }

    static Map<String,long[]> corrupt(long s, long pid,long o,GraphDatabaseService db, String dataset){
        return corrupt( s, pid,o,db,-1,null,dataset);
    }
    static Map<String,Set<Long>> entities=new HashMap<>();
    // dataset=="train" "valid" "test"
    static Map<String,long[]> corrupt(long s, long pid,long o,GraphDatabaseService db, int negativeRate, Random rand, String dataset) {
        Map<String,Object> param=new HashMap<>();
        switch (dataset) {
            case "train":
                param.put("dataset", new String[]{"Training"});
                break;
            case "valid":
                param.put("dataset", new String[]{"Training", "Validation"});
                break;
            case "test":
                param.put("dataset", new String[]{"Training", "Validation", "Test"});
                break;
            default:
                System.out.println("invalid dataset");
                return null;
        }
        Result rs;
        if(!entities.containsKey(dataset)){
            rs=db.execute("match (n)-[p]-() where p.split in $dataset return collect(id(n)) as nids",param);
            entities.put(dataset,new HashSet<>((ArrayList<Long>)rs.next().get("nids")));
        }

        // corrupt subject
        Set<Long> all_subjects=new HashSet<>(entities.get(dataset));
        long[] temp_subject=(long[])db.getNodeById(o).getProperty("subjects_"+pid+"_"+dataset);
        all_subjects.removeAll(LongStream.of(temp_subject).boxed().collect(Collectors.toList()));
        long[] corrupt_subject = ArrayUtils.toPrimitive(all_subjects.toArray(new Long[0]));
        // corrupt object
        Set<Long> all_objects=new HashSet<>(entities.get(dataset));
        long[] temp_object=(long[])db.getNodeById(s).getProperty("objects_"+pid+"_"+dataset);
        all_objects.removeAll(LongStream.of(temp_object).boxed().collect(Collectors.toList()));
        long[] corrupt_object = ArrayUtils.toPrimitive(all_objects.toArray(new Long[0]));

        if(dataset.equals("train")){
            int n_subject_corrupt=rand.nextInt(Math.min(negativeRate, corrupt_subject.length)+1);
            int n_object_corrupt=negativeRate-n_subject_corrupt;
            long[] subject=new long[n_subject_corrupt];
            subject= Arrays.stream(subject).map(item-> corrupt_subject[rand.nextInt(corrupt_subject.length)]).toArray();
            long[] object=new long[n_object_corrupt];
            object= Arrays.stream(object).map(item-> corrupt_object[rand.nextInt(corrupt_object.length)]).toArray();
            return Map.of("corrupt_subject",subject,"corrupt_object",object);
        }else{
            return Map.of("corrupt_subject",corrupt_subject,"corrupt_object",corrupt_object);
        }

    }
    static void gradient_descent(GraphDatabaseService db, long s, long p, long o, long sp, long op,String norm, double alpha, double gamma){
        double[] semb,oemb,spemb,opemb;
        semb = (double[]) db.getNodeById(s).getProperty("embedding");

        if(o==s){
            oemb=semb;
        }else{
            oemb = (double[]) db.getNodeById(o).getProperty("embedding");
        }

        if(sp==s){
            spemb=semb;
        }else if(sp==o){
            spemb=oemb;
        }else{
            spemb = (double[]) db.getNodeById(sp).getProperty("embedding");
        }

        if(op==s){
            opemb=semb;
        }else if(op==o){
            opemb=oemb;
        }else if(op==sp){
            opemb=spemb;
        }else{
            opemb = (double[]) db.getNodeById(op).getProperty("embedding");
        }
        double[] pemb = (double[]) db.getRelationshipById(p).getProperty("embedding");
        double d=getScore(semb,pemb,oemb,norm);
        double dp=getScore(spemb,pemb,opemb,norm);
        double loss=gamma+d-dp;
        if(loss<=0){
            return;
        }
        // gradient descent
        int embedding_dimension = semb.length;
        for (int i = 0; i < embedding_dimension; i++) {
            double x = 2 * (semb[i] + pemb[i] - oemb[i]);
            double xp = 2 * (spemb[i] + pemb[i] - opemb[i]);
            if (norm.equals("l1")) {
                x = Math.signum(x);
                xp = Math.signum(xp);
            }
            pemb[i] += (- alpha * x + alpha * xp);
            semb[i] += (- alpha * x);
            oemb[i] += (alpha * x);
            spemb[i] += (alpha * xp);
            opemb[i] += (- alpha * xp);
        }
        db.getNodeById(s).setProperty("embedding", semb);
        if(o!=s){
            db.getNodeById(o).setProperty("embedding", oemb);
        }
        if(sp!=s && sp!=o){
            db.getNodeById(sp).setProperty("embedding", spemb);
        }
        if(op!=s && op!=o && op!=sp){
            db.getNodeById(op).setProperty("embedding", opemb);
        }
        db.getRelationshipById(p).setProperty("embedding", pemb);
    }
    static double getScore(double[] semb,double[] pemb,double[] oemb, String norm) {
        double score = Double.NaN;
        if (norm.equals("l1")) {
            score = IntStream.range(0, semb.length).mapToDouble(i -> Math.abs(semb[i] + pemb[i] - oemb[i])).sum();
        }
        else if (norm.equals("l2")) {
            score = Math.sqrt(IntStream.range(0, semb.length).mapToDouble(i -> Math.pow(semb[i] + pemb[i] - oemb[i], 2)).sum());
        }
        return score;
    }

    // mode=="stop" or "acc"
    static double evaluate(GraphDatabaseService db,String norm,String mode){
        List<Double> R=new ArrayList<>();
        Map<String,Object> param=new HashMap<>();
        if(mode.equals("stop")){
            param.put("GX",new String[]{"Validation"});
            param.put("GY",new String[]{"Training","Validation"});

        }else if(mode.equals("acc")){
            param.put("GX",new String[]{"Training"});
            param.put("GY",new String[]{"Training","Validation","Test"});
        }else{
            System.out.println("Invalid mode");
            return -1;
        }

        Result rs_gx=db.execute("match ()-[p]->() where p.split in $GX return p as triple",param);
        String[] triple;
        while(rs_gx.hasNext()){
            // [, 5786, -, _derivationally_related_form, 19952]->, 29110]
            triple=rs_gx.next().get("triple").toString().split("[()\\[,]");
            long s=Long.parseLong(triple[1]);
            long p=relation2node.get(triple[3]);
            long o=Long.parseLong(triple[5]);
            double[] sv = (double[]) db.getNodeById(s).getProperty("embedding");
            double[] ov = (double[]) db.getNodeById(o).getProperty("embedding");
            double[] pv = (double[]) db.getRelationshipById(p).getProperty("embedding");
            double posScore=getScore(sv,pv,ov,norm);

            Map<String,long[]> corrupts;
            if(mode.equals("stop")){
                corrupts= corrupt(s, relation2id.get(triple[3]),o,db,"valid");
            }else{
                corrupts= corrupt(s, relation2id.get(triple[3]),o,db,"test");
            }
            // corrupt subject
            int lt = 1;
            int eq = 0;
            long[] corrupt_subject=corrupts.get("corrupt_subject");
            double[] spv;
            for(long sp:corrupt_subject){
                spv = (double[]) db.getNodeById(sp).getProperty("embedding");
                pv = (double[]) db.getRelationshipById(p).getProperty("embedding");
                ov = (double[]) db.getNodeById(o).getProperty("embedding");
                double distance=getScore(spv,pv,ov,norm);
                if(distance<posScore){
                    lt+=1;
                }else if(distance==posScore){
                    eq+=1;
                }
            }
            R.add(lt*1.0 + (eq*1.0/2.0));
            // corrupt object
            lt = 1;
            eq = 0;
            double[] opv;
            long[] corrupt_object=corrupts.get("corrupt_object");
            for(long op:corrupt_object){
                sv = (double[]) db.getNodeById(s).getProperty("embedding");
                pv = (double[]) db.getRelationshipById(p).getProperty("embedding");
                opv = (double[]) db.getNodeById(op).getProperty("embedding");
                double distance=getScore(sv,pv,opv,norm);
                if(distance<posScore){
                    lt+=1;
                }else if(distance==posScore){
                    eq+=1;
                }
            }
            R.add(lt*1.0 + (eq*1.0/2.0));
        }
        // calculate score
        return calculate_MR(R);
    }
    static double calculate_MR(List<Double> R){
        int R_length=R.size();
        return R.stream().mapToDouble(item->item/R_length).sum();
    }

    public static void saveToFile(GraphDatabaseService db, long norm, String outputFile) throws Exception {
        PrintWriter writer = new PrintWriter(new File(outputFile));
        writer.println(norm);

        Transaction tx = db.beginTx();
        writer.println("Entities");
        db.getAllNodes().forEach(n->{
            writer.println(n.getId() + "\t" + Arrays.toString((double[])n.getProperty("embedding")));
        });

        writer.println("Predicates");
        Result res = db.execute("MATCH ()-[p]->() RETURN type(p) AS pred, MIN(id(p)) AS min");
        while (res.hasNext()) {
            Map<String, Object> row = res.next();
            writer.println(row.get("pred") + "\t" + Arrays.toString((double[])db.getRelationshipById((long)row.get("min")).getProperty("embedding")));
        }
        res.close();
        tx.close();
        writer.close();
    }

    public static void saveSplits(GraphDatabaseService db, String outputFile) throws Exception {
        PrintWriter writer = new PrintWriter(new File(outputFile));
        for (String split : new String[] {"Training", "Validation", "Test"}) {
            writer.println(split);
            Transaction tx = db.beginTx();
            Result res = db.execute("MATCH (s)-[p {split:'"+split+"'}]->(o) RETURN id(s) AS s, type(p) AS p, id(o) AS o");
            while (res.hasNext()) {
                Map<String, Object> row = res.next();
                writer.println(row.get("s") + "\t" + row.get("p") + "\t" + row.get("o"));
            }
            res.close();
            tx.close();
        }
        writer.close();
    }
    public static String loadFromFile(GraphDatabaseService db, String inputFile) throws Exception {
        String norm = null;
        Scanner sc = new Scanner(new File(inputFile));
        // Read norm.
        norm = sc.nextLine();
        // Skip entities header.
        sc.nextLine();

        // Load entity embeddings.
        while (sc.hasNextLine()) {
            String line = sc.nextLine();
            if (line.equals("Predicates"))
                break;
            String[] idAndEmbed = line.split("\t");
            db.getNodeById(Long.valueOf(idAndEmbed[0])).setProperty("embedding",Arrays.stream(idAndEmbed[1].substring(1, idAndEmbed[1].length()-1).split(", ")).mapToDouble(x->Double.valueOf(x)).toArray());
        }

        // Load predicate embeddings.
        while (sc.hasNextLine()) {
            String[] predAndEmbed = sc.nextLine().split("\t");
            long relid = (long) db.execute("MATCH ()-[p:`" + predAndEmbed[0] + "`]->() RETURN MIN(id(p)) AS min").next().get("min");
            db.getRelationshipById(relid).setProperty("embedding",
                    Arrays.stream(predAndEmbed[1].substring(1, predAndEmbed[1].length()-1).split(", ")).mapToDouble(x->Double.valueOf(x)).toArray());
        }

        sc.close();

        return norm;
    }
}
